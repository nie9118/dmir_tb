import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
import torch.distributions as D
from torch.nn import functional as F

from functorch import vmap, jacfwd, grad
from torch.autograd.functional import jacobian
from mamba_ssm import Mamba


# "ECL": {
#         "12": "python run_nsts.py    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M      --enc_in 321   --dec_in 321   --c_out 321 "
#               "  --seq_len 36 --label_len 0  --pred_len  12  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 3   --dropout 0   --activation relu   --learning_rate 0.001  --is_bn --zd_kl_weight 0.000001 --zc_kl_weight 0.000001  --hmm_weight 0.000001  ",
#         "24": "python run_nsts.py    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M      --enc_in 321   --dec_in 321   --c_out 321  "
#               "  --seq_len 72 --label_len 0  --pred_len  24  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 3   --dropout 0   --activation relu   --learning_rate 0.001    --is_bn --zd_kl_weight 0.000001 --zc_kl_weight 0.000001  --hmm_weight 0.000001  ",
#         "48": "python run_nsts.py    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M      --enc_in 321   --dec_in 321   --c_out 321  "
#               "  --seq_len 144 --label_len 0  --pred_len  48  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 3   --dropout 0   --activation relu   --learning_rate 0.001  --is_bn --zd_kl_weight 0.000001 --zc_kl_weight 0.000001  --hmm_weight 0.000001  ",
#
#         "72": "python run_nsts.py    --root_path ./dataset/electricity/   --data_path electricity.csv   --data custom   --features M      --enc_in 321   --dec_in 321   --c_out 321  "
#               "   --seq_len 216 --label_len 0  --pred_len  72  --model NSTS  --zd_dim 3  --hidden_dim 512  --hidden_layers 2   --dropout 0   --activation relu   --learning_rate 0.001   --is_bn --zd_kl_weight 0.000001 --zc_kl_weight 0.000001  --hmm_weight 0.000001 ",
#
#     },


class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 var_num,
                 hidden_dim=128,
                 hidden_layers=2,
                 is_bn=False,
                 dropout=0.05,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.var_num = var_num
        if activation == 'gelu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'ide':
            self.activation = nn.Identity()
        else:
            raise NotImplementedError
        if self.hidden_layers == 1:
            self.layers = nn.Sequential(nn.Linear(self.f_in, self.f_out))
        else:
            layers = [nn.Linear(self.f_in, self.hidden_dim),

                      self.activation,
                      nn.Dropout(self.dropout)
                      ]

            for i in range(self.hidden_layers - 2):
                layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                           self.activation,
                           nn.Dropout(dropout)
                           ]
            if is_bn:
                layers += [nn.BatchNorm1d(num_features=self.var_num), nn.Linear(hidden_dim, f_out)]
            else:
                layers += [nn.Linear(hidden_dim, f_out)]
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y


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


# class NPTransitionPrior(nn.Module):
#
#     def __init__(self, lags, latent_size, num_layers=3, hidden_dim=64, compress_dim=10):
#         super().__init__()
#         self.lags = lags
#         self.latent_size = latent_size
#         self.gs = nn.ModuleList([MLP2(input_dim=compress_dim + 1, hidden_dim=hidden_dim,
#                                       output_dim=1, num_layers=num_layers) for _ in
#                                  range(latent_size)]) if latent_size > 100 else nn.ModuleList(
#             [MLP2(input_dim=lags * latent_size + 1, hidden_dim=hidden_dim,
#                   output_dim=1, num_layers=num_layers) for _ in range(latent_size)])
#
#         self.compress = nn.Linear(lags * latent_size, compress_dim)
#         self.compress_dim = compress_dim
#         # self.fc = MLP(input_dim=embedding_dim,hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=2)
#
#     def forward(self, x, mask=None):
#         batch_size, lags_and_length, x_dim = x.shape
#         length = lags_and_length - self.lags
#         # batch_x: (batch_size, lags+length, x_dim) -> (batch_size, length, lags+1, x_dim)
#         batch_x = x.unfold(dimension=1, size=self.lags +
#                                              1, step=1).transpose(2, 3)
#         batch_x = batch_x.reshape(-1, self.lags + 1, x_dim)
#         batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
#         batch_x_t = batch_x[:, -1]  # (batch_size*length, x_dim)
#         # (batch_size*length, lags*x_dim)
#
#         batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
#         if x.shape[-1] > 100:
#             batch_x_lags = self.compress(batch_x_lags)
#         sum_log_abs_det_jacobian = 0
#         residuals = []
#         for i in range(self.latent_size):
#             # (batch_size x length, hidden_dim + lags*x_dim + 1)
#
#             if mask is not None:
#                 batch_inputs = torch.cat(
#                     (batch_x_lags * mask[i], batch_x_t[:, i:i + 1]), dim=-1)
#             else:
#                 batch_inputs = torch.cat(
#                     (batch_x_lags, batch_x_t[:, i:i + 1]), dim=-1)
#
#             residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)
#
#             J = jacfwd(self.gs[i])
#             data_J = vmap(J)(batch_inputs).squeeze()
#             logabsdet = torch.log(torch.abs(data_J[:, -1]))
#
#             sum_log_abs_det_jacobian += logabsdet
#             residuals.append(residual)
#         residuals = torch.cat(residuals, dim=-1)
#         residuals = residuals.reshape(batch_size, length, x_dim)
#
#         log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
#         return residuals, log_abs_det_jacobian


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
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
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


class Encoder_ZC(nn.Module):
    def __init__(self, var_dim, input_len, z_dim, activation,
                 hidden_dim, hidden_layers, dropout, is_bn, zc_kl_weight) -> None:
        super(Encoder_ZC, self).__init__()
        # self.configs = configs
        # latent_size 是啥来的，#HMM跟先验的lags是一个东西吗
        # if configs.enc_in < 100:
        #     self.stationary_transition_prior = NPTransitionPrior(lags=1,
        #                                                          latent_size=var_dim,
        #                                                          num_layers=3,
        #                                                          hidden_dim=128)
        # else:
        self.stationary_transition_prior = NPInstantaneousTransitionPrior(lags=1,
                                                                          latent_size=var_dim,
                                                                          num_layers=3,
                                                                          hidden_dim=3)

        self.zc_rec_net_mean = nn.Sequential(
            MLP(input_len, input_len, var_num=z_dim, activation=activation,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers, dropout=dropout, is_bn=is_bn)
        )

        self.zc_rec_net_std = nn.Sequential(
            MLP(input_len, input_len, var_num=z_dim, activation=activation,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers, dropout=dropout, is_bn=is_bn)
        )

        # self.zc_pred_net_mean = MLP(configs.seq_len, configs.pred_len, var_num=self.configs.zc_dim,
        #                             activation=self.configs.activation,
        #                             hidden_dim=configs.hidden_dim,
        #                             hidden_layers=configs.hidden_layers, dropout=configs.dropout,
        #                             is_bn=self.configs.is_bn)
        #
        # self.zc_pred_net_std = MLP(configs.seq_len, configs.pred_len, var_num=self.configs.zc_dim,
        #                            activation=self.configs.activation,
        #                            hidden_dim=configs.hidden_dim,
        #                            hidden_layers=configs.hidden_layers, dropout=configs.dropout,
        #                            is_bn=self.configs.is_bn)

        self.zc_kl_weight = zc_kl_weight
        self.lags = 1
        self.register_buffer('stationary_dist_mean', torch.zeros(z_dim))
        self.register_buffer('stationary_dist_var', torch.eye(z_dim))

    @property
    def stationary_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.stationary_dist_mean, self.stationary_dist_var)

    def forward(self, x_enc):
        zc_rec_mean = self.zc_rec_net_mean(x_enc.permute(0, 2, 1))
        zc_rec_std = self.zc_rec_net_std(x_enc.permute(0, 2, 1))
        zc_rec = self.reparametrize(zc_rec_mean, zc_rec_std)
        # zc_pred_mean = self.zc_pred_net_mean(zc_rec_mean)
        # zc_pred_std = self.zc_pred_net_std(zc_rec_mean)
        # zc_pred = self.reparametrize(zc_pred_mean, zc_pred_std)

        return zc_rec_mean, zc_rec_std, zc_rec
        # (zc_pred_mean, zc_pred_std, zc_pred)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def kl_loss(self, mus, logvars, z_est):
        lags_and_length = z_est.shape[1]
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

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
        log_qz_laplace = log_qz[:, self.lags:]
        residuals, logabsdet = self.stationary_transition_prior(z_est)
        log_pz_laplace = torch.sum(self.stationary_dist.log_prob(
            residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
                lags_and_length - self.lags)
        kld_laplace = kld_laplace.mean()
        loss = (kld_normal + kld_laplace)
        return loss


class Decoder(nn.Module):
    def __init__(self, configs) -> None:
        super(Decoder, self).__init__()
        self.configs = configs
        self.z_net = nn.Linear(self.configs.zd_dim + self.configs.enc_in, self.configs.enc_in, bias=False)
        self.pred_net = MLP(configs.pred_len, configs.pred_len, var_num=self.configs.enc_in,
                            hidden_dim=configs.hidden_dim,
                            hidden_layers=configs.hidden_layers, is_bn=self.configs.is_bn)

        self.rec_net = MLP(configs.seq_len, configs.seq_len, var_num=self.configs.enc_in,
                           hidden_dim=configs.hidden_dim,
                           hidden_layers=configs.hidden_layers, is_bn=self.configs.is_bn)

        weight = torch.eye(configs.enc_in, self.configs.zd_dim + self.configs.enc_in)

        self.z_net.weight = nn.Parameter(weight)

    def forward(self, zc_rec, zd_rec, zc_pred, zd_pred):
        z_rec = self.z_net(torch.cat([zc_rec, zd_rec], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
        z_pred = self.z_net(torch.cat([zc_pred, zd_pred], dim=1).permute(0, 2, 1)).permute(0, 2, 1)

        x = self.rec_net(z_rec).permute(0, 2, 1)
        y = self.pred_net(z_pred).permute(0, 2, 1)

        return x, y


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
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.configs = configs
        # Embedding

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture

        # class Encoder_ZC(nn.Module):
        #     def __init__(self, lag, var_dim, input_len, pred_len, z_dim, activation,
        #                  hidden_dim, hidden_layers, dropout, is_bn, zc_kl_weight) -> None:

        # self.rec_enc = Encoder_ZC(var_dim=configs.enc_in, input_len=self.seq_len,
        #                           z_dim=configs.enc_in, activation=configs.activation, hidden_dim=10,
        #                           hidden_layers=configs.hidden_layers, dropout=configs.dropout, is_bn=configs.is_bn,
        #                           zc_kl_weight=configs.zc_kl_weight)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.encoder2 = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=5,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=5,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.stationary_transition_prior = NPInstantaneousTransitionPrior(lags=1,
                                                                          latent_size=configs.z_dim,
                                                                          num_layers=3,
                                                                          hidden_dim=128)

        self.alphas_fix = SparsityMatrix(configs.z_dim)
        self.alphas_fix().requires_grad = False
        self.lags = 1
        # self.projector_mean = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        # self.projector_var = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.projector1_mean = nn.Linear(configs.d_model, configs.z_dim, bias=True)
        self.projector1_var = nn.Linear(configs.d_model, configs.z_dim, bias=True)
        self.projector2 = nn.Linear(configs.z_dim, configs.pred_len)

        self.register_buffer('stationary_dist_mean', torch.zeros(self.configs.z_dim))
        self.register_buffer('stationary_dist_var', torch.eye(self.configs.z_dim))

    @property
    def stationary_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.stationary_dist_mean, self.stationary_dist_var)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, is_train):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        # x_mean, x_var, x_enc = self.rec_enc(x_enc)
        # x_enc = x_enc.permute(0, 2, 1)
        # x_mean = x_mean.permute(0, 2, 1)
        # x_var = x_var.permute(0, 2, 1)
        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N
        enc_out_mean = self.projector1_mean(enc_out)
        enc_out_logvar = self.projector1_var(enc_out)
        enc_out = self.reparametrize(mu=enc_out_mean, logvar=enc_out_logvar)
        dec_out = self.projector2(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates
        # dec_out_var = self.projector_var(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
        # dec_out = self.reparametrize(mu=dec_out_mean, logvar=dec_out_var)
        """ 
        print(dec_out_mean.shape)
        print(dec_out_var.shape)
        print(dec_out.shape)
        print(x_var.shape)
        print(x_mean.shape)
        print(x_enc.shape)
        exit()
        """
        # if self.training:
        #    zc_kl_loss = self.rec_enc.kl_loss(mus=torch.cat([x_mean, dec_out_mean], dim=1),
        #                                      logvars=torch.cat([x_var, dec_out_var], dim=1),
        #                                      z_est=torch.cat([x_enc, dec_out], dim=1))
        if is_train:
            kl_loss, sparsity_loss = self.kl_loss(mus=enc_out_mean, logvars=enc_out_logvar, z_est=enc_out)
            pass

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        if is_train:
            return dec_out, kl_loss, sparsity_loss
        else:
            return dec_out

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, is_train=True):
        if is_train:
            dec_out, kl_loss, sparsity_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, is_train)
            return dec_out[:, -self.pred_len:, :], kl_loss, sparsity_loss  # [B, L, D]
        else:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, is_train)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
