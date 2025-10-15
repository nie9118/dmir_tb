import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi
import torch.nn.functional as F
import torch.distributions as D
from torch.func import jacfwd, vmap
# from .mlp import NLayerLeakyMLP, NLayerLeakyNAC

#涛爷的代码

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


# class NPChangeInstantaneousTransitionPrior(nn.Module):
#
#     def __init__(
#             self,
#             lags,
#             latent_size,
#             embedding_dim,
#             num_layers=3,
#             hidden_dim=64):
#         super().__init__()
#         self.L = lags
#         # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))
#         gs = [NLayerLeakyMLP(in_features=hidden_dim + lags * latent_size + 1 + i,
#                              out_features=1,
#                              num_layers=0,
#                              hidden_dim=hidden_dim) for i in range(latent_size)]
#
#         self.gs = nn.ModuleList(gs)
#         self.fc = NLayerLeakyMLP(in_features=embedding_dim,
#                                  out_features=hidden_dim,
#                                  num_layers=2,
#                                  hidden_dim=hidden_dim)
#
#     def forward(self, x, embeddings, alphas):
#         # x: [BS, T, D] -> [BS, T-L, L+1, D]
#         # embeddings: [BS, embed_dims]
#         batch_size, length, input_dim = x.shape
#         embeddings = self.fc(embeddings)
#         embeddings = embeddings.unsqueeze(1).repeat(1, length - self.L, 1).reshape(-1, embeddings.shape[-1])
#         # prepare data
#         x = x.unfold(dimension=1, size=self.L + 1, step=1)
#         x = torch.swapaxes(x, 2, 3)
#         shape = x.shape
#         x = x.reshape(-1, self.L + 1, input_dim)
#         xx, yy = x[:, -1:], x[:, :-1]
#         yy = yy.reshape(-1, self.L * input_dim)
#         # get residuals and |J|
#         residuals = []
#         sum_log_abs_det_jacobian = 0
#         for i in range(input_dim):
#             inputs = torch.cat([xx[:, :, j] * alphas[i][j] for j in range(i)] + [embeddings, yy, xx[:, :, i]], dim=-1)
#             residual = self.gs[i](inputs)
#             with torch.enable_grad():
#                 pdd = vmap(jacfwd(self.gs[i]))(inputs)
#             # Determinant: product of diagonal entries, sum of last entry
#             logabsdet = torch.log(torch.abs(pdd[:, 0, -1]))
#             sum_log_abs_det_jacobian += logabsdet
#             residuals.append(residual)
#
#         residuals = torch.cat(residuals, dim=-1)
#         residuals = residuals.reshape(batch_size, -1, input_dim)
#         sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length - self.L), dim=1)
#         return residuals, sum_log_abs_det_jacobian


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


class MIC(nn.Module):
    """
    MIC layer to extract local and global features
    """

    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, decomp_kernel=[32], conv_kernel=[24],
                 isometric_kernel=[8, 6]):
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

        # return self.norm2(y)
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



class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=zt53IDUR1U
    """

    def __init__(self, configs, conv_kernel=[12, 16]):
        """
        conv_kernel: downsampling and upsampling convolution kernel_size
        """
        super(Model, self).__init__()

        self.configs = configs

        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution
        for ii in conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((configs.seq_len + configs.pred_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((configs.seq_len + configs.pred_len + ii - 1) // ii)

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        # Multiple Series decomposition block from FEDformer
        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # embedding
        self.dec_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding1 = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
        self.conv_trans = SeasonalPrediction(embedding_size=configs.d_model, n_heads=configs.n_heads,
                                             dropout=configs.dropout,
                                             d_layers=configs.d_layers, decomp_kernel=decomp_kernel,
                                             c_out=configs.c_out, conv_kernel=conv_kernel,
                                             isometric_kernel=isometric_kernel)

        self.regression = nn.Linear(configs.seq_len, configs.pred_len)
        self.regression.weight = nn.Parameter(
            (1 / configs.pred_len) * torch.ones([configs.pred_len, configs.seq_len]),
            requires_grad=True)

        configs.z_dim = configs.d_model
        self.rec_criterion = nn.MSELoss()
        self.decoder_dist = 'gaussian'
        self.z_dim_fix = configs.z_dim
        self.alphas_fix = SparsityMatrix(configs.z_dim)
        self.alphas_fix().requires_grad = False
        self.lag = 1
        self.transition_prior_fix = NPInstantaneousTransitionPrior(lags=self.lag,
                                                                   latent_size=configs.z_dim,
                                                                   num_layers=1,
                                                                   hidden_dim=8)
        self.register_buffer('base_dist_mean', torch.zeros(configs.z_dim))
        self.register_buffer('base_dist_var', torch.eye(configs.z_dim))

    @property
    def base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def reconstruction_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'gaussian':
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'sigmoid_gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)

        return recon_loss

    def loss_function(self, x, mus, logvars, zs):
        '''
        VAE ELBO loss: recon_loss + kld_loss (past: N(0,1), future: N(0,1) after flow) + sparsity_loss
        '''
        batch_size, length, _ = x.shape

        # Sparsity loss
        sparsity_loss = 0
        # fix
        if self.z_dim_fix > 0:
            unmasked_alphas_fix = self.alphas_fix()
            mask_fix = (unmasked_alphas_fix > 0.1).float()
            alphas_fix = unmasked_alphas_fix * mask_fix

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(zs)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:, :self.lag]), torch.ones_like(logvars[:, :self.lag]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(zs[:, :self.lag]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:, :self.lag], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()

        # Future KLD
        kld_future = []
        # fix
        if self.z_dim_fix > 0:
            log_qz_laplace = log_qz[:, self.lag:, :self.z_dim_fix]
            residuals, logabsdet, hist_jac = self.transition_prior_fix.forward(zs[:, :, :self.z_dim_fix], alphas_fix)
            log_pz_laplace = torch.sum(self.base_dist.log_prob(residuals), dim=1) + logabsdet
            kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (length - self.lag)
            kld_future.append(kld_laplace)

            trans_show = []
            inst_show = []
            numm = 0
            for jac in hist_jac:
                sparsity_loss = sparsity_loss + F.l1_loss(jac[:, 0, self.lag * self.z_dim_fix:],
                                                          torch.zeros_like(jac[:, 0, self.lag * self.z_dim_fix:]),
                                                          reduction='sum')
                sparsity_loss = sparsity_loss + 10 * F.l1_loss(jac[:, 0, :self.lag * self.z_dim_fix],
                                                               torch.zeros_like(jac[:, 0, :self.lag * self.z_dim_fix]),
                                                               reduction='sum')
                numm = numm + jac.numel()
                trans_show.append(jac[:, 0, :self.lag * self.z_dim_fix].detach().cpu())
                inst_cur = jac[:, 0, self.lag * self.z_dim_fix:].detach().cpu()
                inst_cur = torch.nn.functional.pad(inst_cur, (0, self.z_dim_fix - inst_cur.shape[1], 0, 0),
                                                   mode='constant', value=0)
                inst_show.append(inst_cur)
            # trans_show = torch.stack(trans_show, dim=1).abs().mean(dim=0)
            # inst_show = torch.stack(inst_show, dim=1).abs().mean(dim=0)
            sparsity_loss = sparsity_loss / numm

            # self.trans_show = trans_show
            # self.inst_show = inst_show
        # change
        # if self.z_dim_change>0:
        #     assert(0)
        kld_future = torch.cat(kld_future, dim=-1)
        kld_future = kld_future.mean()

        return sparsity_loss, kld_normal, kld_future

    def forward(self, x_enc, y_enc=None,x_mark_dec=None, is_train=True, is_out_u=False, c_est=None):
        # Multi-scale Hybrid Decomposition
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)

        # embedding
        zeros = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]], device=x_enc.device)
        seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.seq_len:, :], zeros], dim=1)

        dec_out_mean = self.dec_embedding(seasonal_init_dec, x_mark_dec)
        dec_out_std = self.dec_embedding1(seasonal_init_dec, x_mark_dec)

        if is_train:
            dec_out = self.reparametrize(dec_out_mean, dec_out_std)
            xy = self.conv_trans(dec_out)
            assert xy.shape[1] == (self.seq_len + self.pred_len)
            y = xy[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
            y = y[:, -self.pred_len:, :]

            # sparsity_loss, kld_normal, kld_future = self.loss_function(xy, dec_out_mean, dec_out_std, dec_out)

            # Recon loss
            x = xy[:, :self.seq_len, :]
            recon_loss = self.reconstruction_loss(x_enc[:, :self.lag], x[:, :self.lag], self.decoder_dist) + \
                         (self.reconstruction_loss(x_enc[:, self.lag:], x[:, self.lag:], self.decoder_dist)) / (
                                 x_enc.shape[1] - self.lag)

            # other_loss = self.configs.sparsity_weight * sparsity_loss + self.configs.recon_weight * recon_loss + \
            #              self.configs.kld_weight * kld_normal + self.configs.kld_weight * kld_future

            other_loss = self.configs.rec_weight * recon_loss

            return y, other_loss
        else:
            dec_out = dec_out_mean
            xy = self.conv_trans(dec_out)
            assert xy.shape[1] == (self.seq_len + self.pred_len)
            y = xy[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
            y = y[:, -self.pred_len:, :]

            return y, 0

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
