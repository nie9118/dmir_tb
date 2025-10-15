import torch
from torch import nn

import torch.fft as fft


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.mse = nn.MSELoss()
        self.Linear_invarient = nn.Linear(self.configs.seq_len, self.configs.pred_len)
        self.Linear_varient = nn.Linear(self.configs.seq_len, self.configs.pred_len)
        self.top_k = configs.top_k

    def forward(self, x, batch_x_mark, dec_inp, batch_y_mar):
        y = self.forward_lookback(x)

        return y

    def forward_lookback(self, x):
        # torch.isin,torch.eq...,torch.where,torch.gather方法
        # 分不分效果都差不多
        # invarient_x, varient_x = self.get_invarient_varient(x, self.top_k)
        # # 学习结构的时候，这个不变的因果结构还得是用X,如果是用不变的话，在预测的递回归的过程中，因为长度太短，根据不了周期项，提取不变的。
        #
        # invarient_x, varient_x = invarient_x.permute(
        #     0, 2, 1), varient_x.permute(0, 2, 1)
        # invarient_y = self.Linear_invarient(invarient_x)
        # varient_y = self.Linear_invarient(varient_x)
        # invarient_y, varient_y = invarient_y.permute(0, 2, 1), varient_y.permute(0, 2, 1)
        # y = invarient_y + varient_y
        #
        x = x.permute(0, 2, 1)
        y = self.Linear_invarient(x).permute(0, 2, 1)

        return y

    def get_invarient_varient(self, x, k):
        fft_result = fft.fft(x, dim=1)

        amplitude_spectrum = torch.mean(torch.abs(fft_result), dim=-1)
        top_k_values, top_k_indices = torch.topk(amplitude_spectrum, k=k, dim=-1)
        other = torch.ones_like(amplitude_spectrum, dtype=torch.bool)
        other = torch.scatter(other, dim=-1, index=top_k_indices, value=False)
        other = other.unsqueeze(dim=2).repeat(1, 1, self.configs.enc_in)
        top_k_indices = top_k_indices.unsqueeze(dim=2).repeat(1, 1, self.configs.enc_in)
        invarient = torch.zeros_like(fft_result)
        varient = torch.zeros_like(fft_result)

        invarient = torch.scatter(invarient, dim=1, index=top_k_indices,
                                  src=torch.gather(fft_result, dim=1, index=top_k_indices))
        varient[other] = fft_result[other]

        invarient = fft.ifft(invarient, dim=1).real
        varient = fft.ifft(varient, dim=1).real
        return invarient, varient
