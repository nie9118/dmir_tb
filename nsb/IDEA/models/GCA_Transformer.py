import torch
import torch.nn as nn
from models.Autoformer import Model as Autoformer
from models.GCA import Model as GCA


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.autoformer = Autoformer(configs)
        self.gca = GCA(configs)
        self.mse = nn.MSELoss()

    def forward_gca(self, batch_x):
        return self.gca(batch_x)

    def forward_autoformer(self, x, x_time_enc_emb, x_dec, x_time_dec_emb, mask=None):
        dec_out = self.autoformer.forecast(x, x_time_enc_emb, x_dec, x_time_dec_emb)
        dec_out = dec_out[:, -self.configs.pred_len:, :]
        A = self.gca.getStruct_from_encoder(x[:, -self.configs.order:, :])
        gca_in = torch.cat([x[:, -self.configs.order:, :], dec_out[:, :-1, :]], dim=1)
        length = gca_in.shape[1] - self.configs.order + 1
        index_x = torch.arange(self.configs.order).repeat(
            length, 1) + torch.arange(
            length).view(-1, 1)
        gca_in = gca_in[:, index_x, :].view(-1, self.configs.order, gca_in.shape[-1])
        A = A.repeat(1, length, 1, 1).reshape(-1, A.shape[-3],A.shape[-2],A.shape[-1])
        gca_out = self.gca.getPred_from_decoder(gca_in, A)
        gca_out = gca_out.reshape(x.shape[0], length, -1)
        d = self.caculate_distence(gca_out, dec_out)
        return dec_out, d

    def caculate_distence(self, gca_out, dec_out):
        d_losss = self.mse(gca_out.detach(), dec_out)
        return d_losss*self.configs.gca_weight

        # [B, L, D]
