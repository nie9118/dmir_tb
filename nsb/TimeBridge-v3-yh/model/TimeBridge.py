import torch
import torch.nn as nn
from layers.Embed import PatchEmbed
from layers.SelfAttention_Family import TSMixer, ResAttention
from layers.Transformer_EncDec import TSEncoder, IntAttention, PatchSampling, CointAttention


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.revin = configs.revin  # long-term with temporal

        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_p = self.seq_len // self.period
        if configs.num_p is None:
            configs.num_p = self.num_p

        self.embedding = PatchEmbed(configs, num_p=self.num_p)
        self.embedding1 = PatchEmbed(configs, num_p=self.num_p)

        layers = self.layers_init(configs)
        self.encoder = TSEncoder(layers)
        self.encoder1 = TSEncoder(layers)

        out_p = self.num_p if configs.pd_layers == 0 else configs.num_p
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )
        self.decoder1 = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * configs.d_model, configs.pred_len, bias=False)
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(self.c_in, self.c_in),
        )
        with torch.no_grad():
            # 直接使用torch.eye创建单位矩阵
            self.final_mlp[0].weight.data = torch.eye(self.c_in)

            # 将偏置初始化为零
            self.final_mlp[0].bias.data.zero_()

    def layers_init(self, configs):
        integrated_attention = [IntAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout, stable_len=configs.stable_len,
            activation=configs.activation, stable=True, enc_in=self.c_in
        ) for i in range(configs.ia_layers)]

        patch_sampling = [PatchSampling(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=False, stable_len=configs.stable_len,
            in_p=self.num_p if i == 0 else configs.num_p, out_p=configs.num_p,
            dropout=configs.dropout, activation=configs.activation
        ) for i in range(configs.pd_layers)]

        cointegrated_attention = [CointAttention(
            TSMixer(ResAttention(attention_dropout=configs.attn_dropout),
                    configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout,
            activation=configs.activation, stable=False, enc_in=self.c_in, stable_len=configs.stable_len,
        ) for i in range(configs.ca_layers)]

        return [*integrated_attention, *patch_sampling, *cointegrated_attention]

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros((*x_enc.shape[:-1], 4), device=x_enc.device)

        mean, std = (x_enc.mean(1, keepdim=True).detach(),
                     x_enc.std(1, keepdim=True).detach())
        x_enc = (x_enc - mean) / (std + 1e-5)

        # x_enc = self.embedding(x_enc, x_mark_enc)
        # enc_out = self.encoder(x_enc)[0][:, :self.c_in, ...]

        x_mean = self.embedding(x_enc, x_mark_enc)
        x_mean = self.encoder(x_mean)[0][:, :self.c_in, ...]
        x_mean = self.decoder(x_mean).transpose(-1, -2)

        x_std = self.embedding1(x_enc, x_mark_enc)
        x_std = self.encoder1(x_std)[0][:, :self.c_in, ...]
        x_std = self.decoder1(x_std).transpose(-1, -2)

        dec_out = self.reparametrize(x_mean, x_std)

        dec_out = self.final_mlp(dec_out)

        # dec_out = self.decoder(enc_out).transpose(-1, -2)
        # dec_out = self.final_mlp(dec_out)

        return dec_out * std + mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
