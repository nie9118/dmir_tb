import numpy as np
import torch
import torch.nn as nn

import torch.distributions as D

from functorch import vmap, jacfwd, grad
from torch.autograd.functional import jacobian
from einops import rearrange
from layers.StandardNorm import Normalize

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


# 集成了patch_embding 跟tranformer的encoder。根据参数 No_encoder选用要不要encoder

class BackBone(nn.Module):
    def __init__(self, configs):
        super(BackBone, self).__init__()
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.configs = configs
        self.enc_embedding = Embedding_Net(configs.patch_size, configs.seq_len, configs.d_model, configs.emb_dim)

    def forward(self, x_enc):
        enc_out = self.enc_embedding(x_enc).permute(0, 2, 1)
        if not self.configs.No_encoder:
            enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return enc_out


class LinearUnitInit(nn.Linear):
    def reset_parameters(self):
        nn.init.eye_(self.weight)  # 初始化权重为单位矩阵
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)


class Embedding_Net(nn.Module):

    def __init__(self, patch_size, input_len, out_len, emb_dim) -> None:
        super().__init__()
        self.patch_size = patch_size if patch_size <= input_len else input_len
        self.stride = self.patch_size // 2
        self.out_len = out_len

        self.num_patches = int((input_len - self.patch_size) / self.stride + 1)

        self.net1 = MLP(self.patch_size, emb_dim, hidden_layers=1)
        self.net2 = MLP(emb_dim * self.num_patches, self.out_len, hidden_layers=1)

    def forward(self, x):
        B, L, M = x.shape
        if self.num_patches != 1:
            x = rearrange(x, 'b l m -> b m l')
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            x = rearrange(x, 'b m n p -> (b m) n p')
        else:
            x = rearrange(x, 'b l m -> (b m) 1 l')
        x = self.net1(x)
        outputs = self.net2(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b  l m', b=B)
        return outputs


class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 var_num=None,
                 hidden_dim=128,
                 hidden_layers=1,
                 is_bn=False,
                 dropout=0.05,
                 activation='leaky_relu'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.var_num = var_num
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.Identity()

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


class MyHMM(nn.Module):
    def __init__(self, n_class, lags, x_dim, hidden_dim, mode="mle_scaled:H", num_layers=3) -> None:
        super().__init__()
        self.mode, self.feat = mode.split(":")

        self.initial_prob = torch.nn.Parameter(torch.ones(n_class) / n_class, requires_grad=True)
        self.transition_matrix = torch.nn.Parameter(torch.ones(n_class, n_class) / n_class, requires_grad=True)
        self.observation_means = torch.nn.Parameter(torch.rand(n_class, x_dim), requires_grad=True)
        mask = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]])

        self.mask = torch.tensor(mask)

        self.observation_stddevs = torch.nn.Parameter(torch.rand(n_class, x_dim), requires_grad=True)

        if self.mode == "em":
            self.register_buffer('log_A', torch.randn(n_class, n_class))
            self.register_buffer('log_pi', torch.randn(n_class))
        elif self.mode == "mle_scaled" or self.mode == "mle":
            self.log_A = nn.Parameter(torch.randn(n_class, n_class))

            self.log_pi = nn.Parameter(torch.randn(n_class))
        else:
            raise ValueError("mode must be em or mle_scaled or mle, but got {}".format(self.mode))
        self.n_class = n_class
        self.x_dim = x_dim
        self.lags = lags
        if self.feat == "Ht":
            self.trans = MLP(f_in=(lags + 1) * x_dim, hidden_dim=hidden_dim,
                             f_out=n_class * 2 * x_dim, hidden_layers=num_layers)
        elif self.feat == "H":
            self.trans = MLP(f_in=(1) * x_dim, hidden_dim=hidden_dim,
                             f_out=n_class * 2 * x_dim, hidden_layers=num_layers)
        else:
            raise ValueError("feat must be Ht or H")

    def forward_log(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        # length = lags_and_length - self.lags
        log_alpha = torch.zeros(batch_size, length, self.n_class, device=logp_x_c.device)
        log_A = torch.log_softmax(self.log_A, dim=1)
        log_pi = torch.log_softmax(self.log_pi, dim=0)
        for t in range(length):
            if t == 0:
                log_alpha_t = logp_x_c[:, t] + log_pi
            else:
                log_alpha_t = logp_x_c[:, t] + torch.logsumexp(
                    log_alpha[:, t - 1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1)
            log_alpha[:, t] = log_alpha_t
        logp_x = torch.logsumexp(log_alpha[:, -1], dim=-1)

        return logp_x

    def forward_backward_log(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        # length = lags_and_length - self.lags
        log_alpha = torch.zeros(batch_size, length, self.n_class, device=logp_x_c.device)
        log_beta = torch.zeros(batch_size, length, self.n_class, device=logp_x_c.device)
        log_scalers = torch.zeros(batch_size, length, device=logp_x_c.device)
        log_A = torch.log_softmax(self.log_A, dim=1)
        log_pi = torch.log_softmax(self.log_pi, dim=0)
        for t in range(length):
            if t == 0:
                log_alpha_t = logp_x_c[:, t] + log_pi
            else:
                log_alpha_t = logp_x_c[:, t] + torch.logsumexp(
                    log_alpha[:, t - 1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1)
            log_scalers[:, t] = torch.logsumexp(log_alpha_t, dim=-1)
            log_alpha[:, t] = log_alpha_t - log_scalers[:, t].unsqueeze(-1)
        log_beta[:, -1] = torch.zeros(batch_size, self.n_class, device=logp_x_c.device)
        for t in range(length - 2, -1, -1):
            log_beta_t = torch.logsumexp(
                log_beta[:, t + 1].unsqueeze(-1) + log_A.unsqueeze(0) + logp_x_c[:, t + 1].unsqueeze(1), dim=-1)
            log_beta[:, t] = log_beta_t - log_scalers[:, t].unsqueeze(-1)
        log_gamma = log_alpha + log_beta

        logp_x = torch.sum(log_scalers, dim=-1)
        return log_alpha, log_beta, log_scalers, log_gamma, logp_x

    def viterbi_algm(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        log_delta = torch.zeros(batch_size, length, self.n_class, device=logp_x_c.device)
        psi = torch.zeros(batch_size, length, self.n_class, dtype=torch.long, device=logp_x_c.device)

        log_A = torch.log_softmax(self.log_A, dim=1)
        log_pi = torch.log_softmax(self.log_pi, dim=0)

        for t in range(length):
            if t == 0:
                log_delta[:, t] = logp_x_c[:, t] + log_pi
            else:
                max_val, max_arg = torch.max(
                    log_delta[:, t - 1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1)
                log_delta[:, t] = max_val + logp_x_c[:, t]
                psi[:, t] = max_arg

        c = torch.zeros(batch_size, length, dtype=torch.long, device=logp_x_c.device)
        c[:, -1] = torch.argmax(log_delta[:, -1], dim=-1)
        for t in range(length - 2, -1, -1):
            c[:, t] = psi[:, t + 1].gather(1, c[:, t + 1].unsqueeze(1)).squeeze()
        return c  # , logp_x

    def forward(self, x):
        batch_size, lags_and_length, _ = x.shape
        length = lags_and_length - self.lags

        dist = D.Normal(self.observation_means[:, :4], torch.relu(self.observation_stddevs[:, :4]) + 1e-1)

        logp_x_c = dist.log_prob(x[:, :, :4].unsqueeze(2)).sum(-1)

        if self.mode == "em" or self.mode == "mle_scaled":
            log_alpha, log_beta, log_scalers, log_gamma, logp_x = self.forward_backward_log(logp_x_c)
            if self.mode == "em":
                batch_normalizing_factor = torch.log(torch.tensor(batch_size, device=logp_x_c.device))
                expected_log_pi = log_gamma[:, 0, :] - log_gamma[:, 0, :].logsumexp(dim=-1).unsqueeze(-1)
                expected_log_pi = expected_log_pi.logsumexp(dim=0) - batch_normalizing_factor
                log_A = torch.log_softmax(self.log_A, dim=1)
                log_xi = torch.zeros(batch_size, length - 1, self.n_class, self.n_class, device=logp_x_c.device)
                for t in range(length - 1):  # B,Ct,1 B,1,Ct+1 1,Ct,Ct+1 B,1,Ct+1,
                    log_xi_t = log_alpha[:, t].unsqueeze(-1) + log_beta[:, t + 1].unsqueeze(1) + log_A.unsqueeze(
                        0) + logp_x_c[:, t + 1].unsqueeze(1)
                    log_xi_scalers = torch.logsumexp(log_xi_t, dim=(1, 2), keepdim=True)
                    log_xi[:, t] = log_xi_t - log_xi_scalers
                expected_log_A = torch.logsumexp(log_xi, dim=1) - torch.logsumexp(log_xi, dim=(1, 3)).unsqueeze(-1)
                expected_log_A = expected_log_A.logsumexp(dim=0) - batch_normalizing_factor
                self.log_A = expected_log_A.detach()
                self.log_pi = expected_log_pi.detach()
        elif self.mode == "mle":
            logp_x = self.forward_log(logp_x_c)

        c_est = self.viterbi_algm(logp_x_c)

        return logp_x, c_est

    pass


class Encoder_ZD(nn.Module):
    def __init__(self, configs) -> None:
        super(Encoder_ZD, self).__init__()
        self.configs = configs
        # self.zd_net = nn.MultiheadAttention(embed_dim=self.configs.dynamic_dim, num_heads=self.configs.n_heads)
        self.zd_net = nn.Linear(in_features=self.configs.enc_in, out_features=self.configs.zd_dim)

        self.enc_embedding = nn.Sequential(
            MLP(configs.seq_len, configs.dynamic_dim, var_num=self.configs.enc_in,
                activation=self.configs.activation,
                hidden_dim=configs.hidden_dim,
                hidden_layers=configs.hidden_layers, dropout=configs.dropout, is_bn=self.configs.is_bn)
        )
        self.zd_pred_net_mean = MLP(configs.seq_len, configs.pred_len, var_num=self.configs.enc_in,
                                    activation=self.configs.activation,
                                    hidden_dim=configs.hidden_dim,
                                    hidden_layers=configs.hidden_layers, dropout=configs.dropout)
        self.zd_pred_net_std = MLP(configs.seq_len, configs.pred_len, var_num=self.configs.enc_in,
                                   activation=self.configs.activation,
                                   hidden_dim=configs.hidden_dim,
                                   hidden_layers=configs.hidden_layers, dropout=configs.dropout)
        self.nonstationary_transition_prior = NPChangeTransitionPrior(lags=0,
                                                                      latent_size=self.configs.zd_dim,
                                                                      embedding_dim=self.configs.embedding_dim,
                                                                      num_layers=1,
                                                                      hidden_dim=self.configs.hidden_dim)

        self.zd_rec_net_mean = nn.Sequential(
            MLP(configs.dynamic_dim, configs.seq_len, var_num=self.configs.enc_in, activation=self.configs.activation,
                hidden_dim=configs.hidden_dim,
                hidden_layers=configs.hidden_layers, dropout=configs.dropout)
        )
        self.zd_rec_net_std = nn.Sequential(
            MLP(configs.dynamic_dim, configs.seq_len, var_num=self.configs.enc_in, activation=self.configs.activation,
                hidden_dim=configs.hidden_dim,
                hidden_layers=configs.hidden_layers, dropout=configs.dropout)
        )
        self.register_buffer('nonstationary_dist_mean', torch.zeros(self.configs.zd_dim))
        self.register_buffer('nonstationary_dist_var', torch.eye(self.configs.zd_dim))

    @property
    def nonstationary_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.nonstationary_dist_mean, self.nonstationary_dist_var)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x_enc, is_train=True):
        zd_x_enc = self.enc_embedding(x_enc.permute(0, 2, 1))
        # zd = self.zd_net(zd_x_enc, zd_x_enc, zd_x_enc)[0][:, :self.configs.zd_dim]
        zd = self.zd_net(zd_x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        zd_rec_mean = self.zd_rec_net_mean(zd)
        zd_rec_std = self.zd_rec_net_std(zd)
        zd_rec = self.reparametrize(zd_rec_mean, zd_rec_std) if is_train else zd_rec_mean
        zd_pred_mean = self.zd_pred_net_mean(zd_rec_mean)
        zd_pred_std = self.zd_pred_net_std(zd_rec_mean)
        zd_pred = self.reparametrize(zd_pred_mean, zd_pred_std) if is_train else zd_pred_mean
        return (zd_rec_mean, zd_rec_std, zd_rec), (zd_pred_mean, zd_pred_std, zd_pred)

    def kl_loss(self, mus, logvars, z_est, c_embedding):
        lags_and_length = z_est.shape[1]
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Future KLD
        log_qz_laplace = log_qz
        residuals, logabsdet = self.nonstationary_transition_prior.forward(z_est, c_embedding)

        log_pz_laplace = torch.sum(self.nonstationary_dist.log_prob(
            residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (
                              torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
                          lags_and_length)
        kld_laplace = kld_laplace.mean()
        loss = kld_laplace
        return loss


class Encoder_ZC(nn.Module):
    def __init__(self, configs) -> None:
        super(Encoder_ZC, self).__init__()
        self.configs = configs
        # latent_size 是啥来的，#HMM跟先验的lags是一个东西吗
        if configs.enc_in < 100:

            self.stationary_transition_prior = NPTransitionPrior(lags=self.configs.lags,
                                                                 latent_size=self.configs.zc_dim,
                                                                 num_layers=1,
                                                                 hidden_dim=self.configs.hidden_dim)
        else:
            self.stationary_transition_prior = NPTransitionPrior(lags=self.configs.lags,
                                                                 latent_size=self.configs.zc_dim,
                                                                 num_layers=1,
                                                                 hidden_dim=3)

        self.net = Embedding_Net(configs.patch_size, configs.seq_len, configs.seq_len, configs.emb_dim)

        self.feature_extractor = BackBone(configs)
        self.zc_rec_net_mean = MLP(configs.d_model, configs.seq_len, var_num=self.configs.zc_dim,
                                   activation=self.configs.activation,
                                   hidden_dim=configs.hidden_dim,
                                   hidden_layers=configs.hidden_layers, dropout=configs.dropout,
                                   is_bn=self.configs.is_bn)

        self.zc_rec_net_std = MLP(configs.d_model, configs.seq_len, var_num=self.configs.zc_dim,
                                  activation=self.configs.activation,
                                  hidden_dim=configs.hidden_dim,
                                  hidden_layers=configs.hidden_layers, dropout=configs.dropout,
                                  is_bn=self.configs.is_bn)

        self.zc_pred_net_mean = MLP(configs.seq_len, configs.pred_len, var_num=self.configs.zc_dim,
                                    activation=self.configs.activation,
                                    hidden_dim=configs.hidden_dim,
                                    hidden_layers=configs.hidden_layers, dropout=configs.dropout,
                                    is_bn=self.configs.is_bn)

        self.zc_pred_net_std = MLP(configs.seq_len, configs.pred_len, var_num=self.configs.zc_dim,
                                   activation=self.configs.activation,
                                   hidden_dim=configs.hidden_dim,
                                   hidden_layers=configs.hidden_layers, dropout=configs.dropout,
                                   is_bn=self.configs.is_bn)

        self.zc_kl_weight = configs.zc_kl_weight
        self.lags = self.configs.lags
        self.register_buffer('stationary_dist_mean', torch.zeros(self.configs.zc_dim))
        self.register_buffer('stationary_dist_var', torch.eye(self.configs.zc_dim))

    @property
    def stationary_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.stationary_dist_mean, self.stationary_dist_var)

    def forward(self, x_enc, is_train=True):
        feature = self.feature_extractor(x_enc)

        zc_rec_mean = self.zc_rec_net_mean(feature)
        zc_rec_std = self.zc_rec_net_std(feature)
        zc_rec = self.reparametrize(zc_rec_mean, zc_rec_std) if is_train else zc_rec_mean
        zc_pred_mean = self.zc_pred_net_mean(zc_rec_mean)
        zc_pred_std = self.zc_pred_net_std(zc_rec_mean)

        zc_pred = self.reparametrize(zc_pred_mean, zc_pred_std) if is_train else zc_pred_mean

        return (zc_rec_mean, zc_rec_std, zc_rec), (zc_pred_mean, zc_pred_std, zc_pred)

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
        kld_laplace = (
                              torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
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


class NPTransitionPrior(nn.Module):

    def __init__(self, lags, latent_size, num_layers=3, hidden_dim=64, compress_dim=10):
        super().__init__()
        self.lags = lags
        self.latent_size = latent_size

        self.gs = nn.ModuleList(
            [MLP(f_in=compress_dim + 1, f_out=1, hidden_dim=hidden_dim, hidden_layers=num_layers) for _ in
             range(latent_size)]) if latent_size > 100 else nn.ModuleList(
            [MLP(f_in=lags * latent_size + 1, f_out=1, hidden_dim=hidden_dim, hidden_layers=num_layers) for _ in
             range(latent_size)])

        self.compress = nn.Linear(lags * latent_size, compress_dim)
        self.compress_dim = compress_dim

    def forward(self, x, mask=None):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags
        # batch_x: (batch_size, lags+length, x_dim) -> (batch_size, length, lags+1, x_dim)
        batch_x = x.unfold(dimension=1, size=self.lags +
                                             1, step=1).transpose(2, 3)
        batch_x = batch_x.reshape(-1, self.lags + 1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
        batch_x_t = batch_x[:, -1]  # (batch_size*length, x_dim)
        # (batch_size*length, lags*x_dim)

        batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
        if x.shape[-1] > 100:
            batch_x_lags = self.compress(batch_x_lags)
        sum_log_abs_det_jacobian = 0
        residuals = []
        for i in range(self.latent_size):
            # (batch_size x length, hidden_dim + lags*x_dim + 1)

            if mask is not None:
                batch_inputs = torch.cat(
                    (batch_x_lags * mask[i], batch_x_t[:, i:i + 1]), dim=-1)
            else:
                batch_inputs = torch.cat(
                    (batch_x_lags, batch_x_t[:, i:i + 1]), dim=-1)

            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            J = jacfwd(self.gs[i])
            data_J = vmap(J)(batch_inputs).squeeze()
            logabsdet = torch.log(torch.abs(data_J[:, -1]))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)

        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian


class NPChangeTransitionPrior(nn.Module):

    def __init__(
            self,
            lags,
            latent_size,
            embedding_dim,
            num_layers=3,
            hidden_dim=64):
        super().__init__()
        self.latent_size = latent_size
        self.lags = lags
        self.gs = nn.ModuleList([MLP(f_in=embedding_dim + 1, hidden_dim=hidden_dim,
                                     f_out=1, hidden_layers=num_layers) for _ in range(latent_size)])

        self.gs = nn.ModuleList([MLP(f_in=embedding_dim + 1, hidden_dim=hidden_dim,
                                     f_out=1, hidden_layers=num_layers) for _ in range(latent_size)])

    def forward(self, x, embeddings):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags

        batch_x = x.unfold(dimension=1, size=self.lags +
                                             1, step=1).transpose(2, 3)

        batch_embeddings = embeddings[:, -length:].expand(batch_size, length, -1).reshape(batch_size * length, -1)
        batch_x = batch_x.reshape(-1, self.lags + 1, x_dim)

        batch_x_t = batch_x[:, -1:]  # (batch_size*length, x_dim)

        sum_log_abs_det_jacobian = 0
        residuals = []
        for i in range(self.latent_size):
            batch_inputs = torch.cat(
                (batch_embeddings, batch_x_t[:, :, i]), dim=-1)

            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            J = jacfwd(self.gs[i])
            data_J = vmap(J)(batch_inputs).squeeze()
            logabsdet = torch.log(torch.abs(data_J[:, -1]))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super(Model, self).__init__()
        self.configs = configs
        self.configs.zc_dim = self.configs.enc_in
        self.encoder_zd = Encoder_ZD(configs)
        self.encoder_zc = Encoder_ZC(configs)
        self.decoder = Decoder(configs)
        self.encoder_u = MyHMM(n_class=self.configs.n_class, lags=0,
                               x_dim=self.configs.enc_in, hidden_dim=self.configs.hidden_dim, mode="mle_scaled:H")
        self.c_embeddings = nn.Embedding(configs.n_class, configs.embedding_dim)
        self.normalize = Normalize(num_features=configs.enc_in, affine=True, non_norm=not self.configs.is_norm)
        self.rec_criterion = nn.MSELoss()

    def forward(self, x_enc, y_enc=None, x_mark_dec=None, is_train=True, is_out_u=False, c_est=None):
        x_enc = self.normalize(x_enc, "norm")

        (zd_rec_mean, zd_rec_std, zd_rec), (zd_pred_mean, zd_pred_std, zd_pred) = self.encoder_zd(x_enc, is_train)
        (zc_rec_mean, zc_rec_std, zc_rec), (zc_pred_mean, zc_pred_std, zc_pred) = self.encoder_zc(x_enc, is_train)
        x, y = self.decoder(zc_rec, zd_rec, zc_pred, zd_pred)

        y = self.normalize(y, "denorm")

        other_loss = self.rec_criterion(x, x_enc) * self.configs.rec_weight

        if is_train and (not self.configs.No_prior):
            y_enc = (y_enc - mean_enc) / std_enc
            hmm_loss = 0
            if c_est == None:
                E_logp_x, c_est = self.encoder_u(torch.cat([x_enc, y_enc], dim=1))
                hmm_loss = -E_logp_x.mean()
            embeddings = self.c_embeddings(c_est)

            zc_kl_loss = self.encoder_zc.kl_loss(torch.cat([zc_rec_mean, zc_pred_mean], dim=2).permute(0, 2, 1),
                                                 torch.cat([zc_rec_std, zc_pred_std], dim=2).permute(0, 2, 1),
                                                 torch.cat([zc_rec, zc_pred], dim=2).permute(0, 2, 1))
            zd_kl_loss = self.encoder_zd.kl_loss(torch.cat([zd_rec_mean, zd_pred_mean], dim=2).permute(0, 2, 1),
                                                 torch.cat([zd_rec_std, zd_pred_std], dim=2).permute(0, 2, 1),
                                                 torch.cat([zd_rec, zd_pred], dim=2).permute(0, 2, 1), embeddings)
            other_loss = zc_kl_loss * self.configs.zc_kl_weight + zd_kl_loss * self.configs.zd_kl_weight + hmm_loss * self.configs.hmm_weight + other_loss
            if is_out_u:
                return y, other_loss, c_est
        return y, other_loss
