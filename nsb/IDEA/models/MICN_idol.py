import torch
import torch.nn as nn
from torch import vmap

from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi
import torch.nn.functional as F
import torch.distributions as D


class MLP2(nn.Module):
    """A simple MLP with ReLU activations"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, leaky_relu_slope=0.2):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """

    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24],
                 isometric_kernel=[18, 6]):
        super(MIC, self).__init__()
        self.conv_kernel = conv_kernel

        # isometric convolution
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in isometric_kernel])

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in conv_kernel])

        # upsampling convolution
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in conv_kernel])

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = torch.nn.Conv2d(in_channels=feature_size, out_channels=feature_size,
                                     kernel_size=(len(self.conv_kernel), 1))

        # feedforward network
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric convolution
        zeros = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1), device=input.device)
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]  # truncate

        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        # multi-scale
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)

            # merge
        mg = torch.tensor([], device=src.device)
        for i in range(len(self.conv_kernel)):
            mg = torch.cat((mg, multi[i].unsqueeze(1)), dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)

        return self.norm2(mg + y)


class SeasonalPrediction(nn.Module):
    def __init__(self, embedding_size=512, n_heads=8, dropout=0.05, d_layers=1, decomp_kernel=[32], c_out=1,
                 conv_kernel=[2, 4], isometric_kernel=[18, 6]):
        super(SeasonalPrediction, self).__init__()

        self.mic = nn.ModuleList([MIC(feature_size=embedding_size, n_heads=n_heads,
                                      decomp_kernel=decomp_kernel, conv_kernel=conv_kernel,
                                      isometric_kernel=isometric_kernel)
                                  for i in range(d_layers)])

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)


class BaseModel(nn.Module):
    def __init__(self, enc_in, seq_len, pred_len, d_model, n_heads, dropout, d_layers, c_out, conv_kernel=[12, 16],
                 ):
        super().__init__()
        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution
        self.seq_len=seq_len
        self.pred_len=pred_len
        for ii in conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((seq_len + pred_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((seq_len + pred_len + ii - 1) // ii)

        self.dec_embedding = DataEmbedding(enc_in, d_model, dropout=dropout)

        # Multiple Series decomposition block from FEDformer
        self.decomp_multi = series_decomp_multi(decomp_kernel)
        self.conv_trans = SeasonalPrediction(embedding_size=d_model, n_heads=n_heads,
                                             dropout=dropout,
                                             d_layers=d_layers, decomp_kernel=decomp_kernel,
                                             c_out=c_out, conv_kernel=conv_kernel,
                                             isometric_kernel=isometric_kernel)
        self.regression = nn.Linear(seq_len, pred_len)
        self.regression.weight = nn.Parameter(
            (1 / pred_len) * torch.ones([pred_len, seq_len]),
            requires_grad=True)
        self.mean_and_std_net=nn.Linear(1,2)

    def forward(self, x_enc):
        # Multi-scale Hybrid Decomposition
        seasonal_init_enc,_ = self.decomp_multi(x_enc)
        # trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)

        # embedding
        zeros = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]], device=x_enc.device)
        seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.seq_len:, :], zeros], dim=1)
        dec_out = self.dec_embedding(seasonal_init_dec, None)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len:, :]
        # dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
        mean,std=torch.chunk(self.mean_and_std_net(torch.unsqueeze(dec_out,dim=-1)),dim=-1,chunks=2)

        return torch.squeeze(mean),torch.squeeze(std)


class NPInstantaneousTransitionPrior(nn.Module):

    def __init__(
            self,
            lags,
            latent_size,
            num_layers=3,
            hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))
        # (input_dim=compress_dim + 1, hidden_dim=hidden_dim,
        # #                                       output_dim=1, num_layers=num_layers)
        self.lags = lags
        self.latent_size = latent_size
        gs = [MLP2(input_dim=lags * latent_size + 1 + i,
                   output_dim=1,
                   num_layers=num_layers,
                   hidden_dim=hidden_dim) for i in range(latent_size)]

        self.gs = nn.ModuleList(gs)

    def forward(self, x, alphas):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # prepare data
        x = x.unfold(dimension=1, size=self.L + 1, step=1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L + 1, input_dim)
        xx, yy = x[:, -1:], x[:, :-1]
        yy = yy.reshape(-1, self.L * input_dim)
        # get residuals and |J|
        residuals = []

        hist_jac = []

        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            inputs = torch.cat([yy] + [xx[:, :, j] * alphas[i][j] for j in range(i)] + [xx[:, :, i]], dim=-1)
            # inputs = torch.cat([yy[:, :, i]] + [xx[:,:,j] * alphas[i][j] for j in range(i)] + [xx[:,:,i]], dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = vmap(torch.func.jacfwd(self.gs[i]))(inputs)
            # Determinant: product of diagonal entries, sum of last entry
            logabsdet = torch.log(torch.abs(pdd[:, 0, -1]))

            # hist_jac.append(torch.unsqueeze(pdd[:,0,:self.L*input_dim], dim=1))
            hist_jac.append(torch.unsqueeze(pdd[:, 0, :-1], dim=1))
            # hist_jac.append(torch.unsqueeze(torch.abs(pdd[:,0,:self.L*input_dim]), dim=1))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        # hist_jac = torch.cat(hist_jac, dim=1) # BS * input_dim * (L * input_dim)
        # hist_jac = torch.mean(hist_jac, dim=0) # input_dim * (L * input_dim)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length - self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac

class SparsityMatrix(nn.Module):
    def __init__(self, z_dim):
        super(SparsityMatrix, self).__init__()

        # prior_matrix = [[0, 0, 0],
        #                 [1, 0, 0],
        #                 [0, 1, 0]]
        prior_matrix = torch.zeros((z_dim, z_dim)).cuda()
        for i in range(z_dim - 1):
            prior_matrix[i + 1, i] = 1
            pass
        prior_matrix = torch.tensor(prior_matrix)
        self.trainable_parameters = nn.Parameter(prior_matrix)

        # self.trainable_parameters = nn.Parameter((torch.ones([z_dim, z_dim])))

    def forward(self):
        return self.trainable_parameters

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        configs.z_dim=configs.enc_in
        self.configs = configs

        self.z_rec_encoder = BaseModel(configs.enc_in, configs.seq_len, configs.seq_len, configs.d_model,
                                       configs.n_heads, configs.dropout, configs.d_layers, c_out=configs.z_dim )
        self.z_pred_encoder = BaseModel(configs.z_dim, configs.seq_len, configs.pred_len, configs.d_model // 2,
                                        configs.n_heads, configs.dropout, configs.d_layers, c_out=configs.z_dim)
        self.decoder = nn.Sequential(nn.Linear(configs.z_dim, configs.c_out))

        self.rec_criterion = nn.MSELoss()
        self.alphas_fix = SparsityMatrix(configs.z_dim)
        self.alphas_fix().requires_grad = False
        self.lags=1


        self.stationary_transition_prior = NPInstantaneousTransitionPrior(lags=1,
                                                                          latent_size=configs.z_dim,
                                                                          num_layers=1,
                                                                          hidden_dim=8)
        self.register_buffer('stationary_dist_mean', torch.zeros(self.configs.z_dim))
        self.register_buffer('stationary_dist_var', torch.eye(self.configs.z_dim))

    @property
    def stationary_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.stationary_dist_mean, self.stationary_dist_var)

    def forward(self, x, is_train=True):
        z_rec_mean, z_rec_std = self.z_rec_encoder(x)
        z_rec = self.reparametrize(z_rec_mean, z_rec_std)
        z_pred_mean, z_pred_std = self.z_pred_encoder(z_rec)
        z_pred = self.reparametrize(z_pred_mean, z_pred_std)
        rec_x = self.decoder(z_rec)
        pred_y = self.decoder(z_pred)

        if not is_train:
            return pred_y,0

        kl_loss, sparsity_loss = self.kl_loss(torch.cat([z_rec_mean, z_pred_mean], dim=1),
                                              torch.cat([z_rec_std, z_pred_std], dim=1),
                                              torch.cat([z_rec, z_pred], dim=1))

        other_loss = kl_loss * self.configs.kl_weight + sparsity_loss * self.configs.sparsity_weight + self.rec_criterion(
            x, rec_x) * self.configs.rec_weight
        return pred_y, other_loss

    def kl_loss(self, mus, logvars, z_est):

        lags_and_length = z_est.shape[1]
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        unmasked_alphas_fix = self.alphas_fix()
        mask_fix = (unmasked_alphas_fix > 0.1).float()
        alphas_fix = unmasked_alphas_fix * mask_fix

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lags:, :self.configs.z_dim]
        residuals, logabsdet, hist_jac = self.stationary_transition_prior.forward(z_est, alphas_fix)

        log_pz_laplace = torch.sum(self.stationary_dist.log_prob(
            residuals), dim=1) + logabsdet
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
                lags_and_length - self.lags)
        kld_laplace = kld_laplace.mean()

        sparsity_loss = 0
        numm = 0
        for jac in hist_jac:
            sparsity_loss = sparsity_loss + F.l1_loss(jac[:, 0, self.lags * self.configs.z_dim:],
                                                      torch.zeros_like(jac[:, 0, self.lags * self.configs.z_dim:]),
                                                      reduction='sum')
            sparsity_loss = sparsity_loss + 5 * F.l1_loss(jac[:, 0, :self.lags * self.configs.z_dim],
                                                          torch.zeros_like(jac[:, 0, :self.lags * self.configs.z_dim]),
                                                          reduction='sum')
            numm = numm + jac.numel()
        sparsity_loss = sparsity_loss / numm

        kl_loss = (kld_normal + kld_laplace)

        return kl_loss, sparsity_loss

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
