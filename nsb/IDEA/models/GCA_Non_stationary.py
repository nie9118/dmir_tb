import torch
from torch import nn

import torch.fft as fft
import torch.nn.functional as F
from torch.distributions import kl_divergence


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.mse = nn.MSELoss()

        modules = [nn.Sequential(nn.Linear(self.configs.num_vars * self.configs.order, self.configs.hidden_layer_size))]
        if self.configs.num_hidden_layer > 1:
            for j in range(self.num_hidden_layer - 1):
                modules.extend(nn.Sequential(nn.Linear(self.configs.hidden_layer_size, self.configs.hidden_layer_size)))
        modules.extend(nn.Sequential(
            nn.BatchNorm1d(self.configs.hidden_layer_size),
            nn.Linear(self.configs.hidden_layer_size, self.configs.num_vars * self.configs.num_vars * 2)))

        self.coeff_net = nn.Sequential(*modules)
        self.emb_net = nn.Sequential(nn.Linear(configs.num_vars, configs.att_emb_dim))

        self.varient_att_net = nn.MultiheadAttention(embed_dim=configs.att_emb_dim, num_heads=configs.att_head,
                                                     batch_first=True)
        self.varient_mlp_net = nn.Sequential(nn.Linear(configs.subset_length * configs.att_emb_dim, 3 * configs.u_dim),
                                             nn.BatchNorm1d(3 * configs.u_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(3 * configs.u_dim, configs.u_dim))
        # 如果有多种实现的话，看效果
        self.koopman = nn.Sequential(nn.Linear(configs.u_dim, 2 * configs.u_dim),
                                     nn.Linear(2 * configs.u_dim, configs.u_dim)
                                     )
        self.effect_net = nn.Sequential(nn.Linear(self.configs.num_vars * self.configs.order + self.configs.u_dim, 1))
        self.predict_net = nn.Sequential(
            nn.Linear(self.configs.num_vars * self.configs.order + self.configs.u_dim, 3 * self.configs.num_vars),
            nn.BatchNorm1d(3 * self.configs.num_vars),
            nn.ReLU(),
            nn.Linear(3 * self.configs.num_vars, self.configs.num_vars)
        )

        self.rec_att = nn.MultiheadAttention(configs.att_emb_dim, configs.att_head, batch_first=True)
        self.rec_emb_net = nn.Sequential(nn.Linear(configs.u_dim, configs.subset_length * configs.att_emb_dim))
        self.rec_net = nn.Sequential(nn.Linear(configs.att_emb_dim, configs.num_vars))

    # 1.推断矩阵的时候，使用matmul，并不是矩阵的mask那样
    # 2.在预测的时候，U,不变的输入都是递归输入的（之前是不变，但是pre-window得不到不变的。因此还是要输入整段x），但是结构是用lookback阶段的最后一组（order个）。所以结构要不要也推断一下。
    # 3.gumbel-softmax 现在是用到硬的，这个怎么调比较好？
    # 4.因果结构对齐的时候，不同lag的结构不一样，但是不同时期同一lag结构一样。

    def forward(self, x):

        pre_select_x, true_select_x, rec_varient, varient_x, select_invarient_structure, all_lag_structures, last_u, last_structure, last_x = self.forward_lookback(
            x, 2)
        predict_y = self.forward_predict(last_x, last_u, last_structure)
        structure_loss = self.calculate_structure_loss(select_invarient_structure, all_lag_structures)
        select_x_loss = self.mse(pre_select_x, true_select_x)
        varient_x_loss = self.mse(rec_varient, varient_x)

        return predict_y, structure_loss, select_x_loss, varient_x_loss

    def forward_lookback(self, x, k):
        # torch.isin,torch.eq...,torch.where,torch.gather方法

        invarient_x, varient_x = self.get_invarient_varient(x, k)
        # 学习结构的时候，这个不变的因果结构还得是用X,如果是用不变的话，在预测的递回归的过程中，因为长度太短，根据不了周期项，提取不变的。
        invarient_x = x
        subset_num = self.configs.seq_len // self.configs.subset_length

        index_x = torch.arange(self.configs.order).repeat(self.configs.subset_length - self.configs.order
                                                          , 1) + torch.arange(
            self.configs.subset_length - self.configs.order).view(-1, 1)
        # 符合广播即可 (3，7，5) ,(3,1,1)也符合广播
        index_x = (index_x.repeat(subset_num, 1, 1) + torch.arange(0, self.configs.seq_len,
                                                                   self.configs.subset_length).view(
            subset_num, 1, 1)).reshape(-1, self.configs.order)

        index_pre_x = (
                torch.arange(self.configs.order, self.configs.subset_length).repeat(subset_num, 1) + torch.arange(0,
                                                                                                                  self.configs.seq_len,
                                                                                                                  self.configs.subset_length).view(
            subset_num, 1)).reshape(-1)
        select_invarient_x = invarient_x[:, index_x, :].view(-1, self.configs.order, self.configs.num_vars)
        select_invarient_structure, all_lag_structures = self.caculate_invarient_structrue(select_invarient_x)
        U, rec_varient = self.caculate_varient(varient_x)
        pre_select_x = self.caculate_effect(select_invarient_x, select_invarient_structure, U).squeeze(dim=-1)
        last_structure = select_invarient_structure.reshape(x.shape[0], -1, self.configs.order, self.configs.num_vars,
                                                            self.configs.num_vars)[:, -1]
        last_x = invarient_x[:, -self.configs.order:, :]
        last_u = U[:, -1]
        true_select_x = x[:, index_pre_x, :]
        pre_select_x = pre_select_x.reshape(true_select_x.shape)

        return pre_select_x, true_select_x, rec_varient, varient_x, select_invarient_structure, all_lag_structures, last_u, last_structure, last_x

    def caculate_effect(self, invarient_x, structure, U, lookback=True):
        invarient = (invarient_x.unsqueeze(dim=-1).repeat(1, 1, 1, self.configs.num_vars) * structure) \
            .permute(0, 3, 1, 2).reshape(-1, self.configs.num_vars, self.configs.order * self.configs.num_vars)
        if lookback:

            U = U.unsqueeze(dim=2).unsqueeze(dim=3).repeat(1, 1, self.configs.subset_length - self.configs.order,
                                                           self.configs.num_vars, 1).reshape(-1, self.configs.num_vars,
                                                                                             self.configs.u_dim)
            total_input = torch.cat([invarient, U], dim=-1)

        else:
            U = U.unsqueeze(dim=1).repeat(1, self.configs.num_vars, 1).reshape(-1, self.configs.num_vars,
                                                                               self.configs.u_dim)
            total_input = torch.cat([invarient, U], dim=-1)
        pre = self.effect_net(total_input).squeeze(dim=-1)
        return pre

    def get_invarient_varient(self, x, k):
        fft_result = fft.fft(x, dim=1)

        amplitude_spectrum = torch.mean(torch.abs(fft_result), dim=-1)
        top_k_values, top_k_indices = torch.topk(amplitude_spectrum, k=k, dim=-1)
        other = torch.ones_like(amplitude_spectrum, dtype=torch.bool)
        other = torch.scatter(other, dim=-1, index=top_k_indices, value=False)
        other = other.unsqueeze(dim=2).repeat(1, 1, self.configs.num_vars)
        top_k_indices = top_k_indices.unsqueeze(dim=2).repeat(1, 1, self.configs.num_vars)
        invarient = torch.zeros_like(fft_result)
        varient = torch.zeros_like(fft_result)

        invarient = torch.scatter(invarient, dim=1, index=top_k_indices,
                                  src=torch.gather(fft_result, dim=1, index=top_k_indices))
        varient[other] = fft_result[other]

        invarient = fft.ifft(invarient, dim=1).real
        varient = fft.ifft(varient, dim=1).real
        return invarient, varient

    # def forward_predict(self, last_x, last_u, last_structure):
    #     subset_num = self.configs.pred_len // self.configs.subset_length
    #     predict = []
    #     now_u = last_u
    #     now_x = last_x
    #     structure = last_structure
    #     for i in range(subset_num):
    #
    #         now_u = self.koopman(now_u)
    #         for j in range(self.configs.subset_length):
    #             pre = self.caculate_effect(now_x, structure, now_u, lookback=False)
    #
    #             predict.append(pre)
    #             now_x[:, :self.configs.order - 1] = now_x[:, 1:self.configs.order]
    #             now_x[:, self.configs.order - 1] = pre
    #
    #     predict = torch.stack(predict, dim=1)
    #     return predict

    def forward_predict(self, last_x, last_u, last_structure):
        predict = []
        now_u = last_u
        now_x = last_x
        structure = last_structure

        for j in range(self.configs.pred_len):
            pre = self.caculate_effect(now_x, structure, now_u, lookback=False)

            predict.append(pre)
            now_x[:, :self.configs.order - 1] = now_x[:, 1:self.configs.order]
            now_x[:, self.configs.order - 1] = pre

        predict = torch.stack(predict, dim=1)
        return predict

    def caculate_varient(self, varient):
        subset_num = varient.shape[1] // self.configs.subset_length
        varient_x = varient.reshape(-1, subset_num, self.configs.subset_length, self.configs.num_vars)
        att_emb = self.emb_net(varient_x[:, 0])
        att_value = self.varient_att_net(att_emb, att_emb, att_emb)[0]
        u1 = self.varient_mlp_net(att_value.reshape(-1, self.configs.subset_length * self.configs.att_emb_dim))
        reconstructure = torch.zeros_like(varient)
        U = []
        now_u = u1
        for i in range(subset_num):
            U.append(now_u)
            sub_varient_x_emb = self.rec_emb_net(now_u).reshape(-1, self.configs.subset_length,
                                                                self.configs.att_emb_dim)

            sub_varient_x_emb = self.rec_att(sub_varient_x_emb, sub_varient_x_emb, sub_varient_x_emb)[0]

            sub_varient_x = self.rec_net(sub_varient_x_emb)
            reconstructure[:, i * self.configs.subset_length:(i + 1) * self.configs.subset_length] = sub_varient_x
            now_u = self.koopman(now_u)

        U = torch.stack(U, dim=1)
        return U, reconstructure

    def caculate_invarient_structrue(self, invarient_x):

        pred_k_structures = torch.zeros(
            [invarient_x.shape[0], self.configs.order, self.configs.num_vars, self.configs.num_vars]).to(
            invarient_x.device)
        all_lag_structures = torch.zeros(
            [invarient_x.shape[0], self.configs.order, self.configs.num_vars, self.configs.num_vars, 2]).to(
            invarient_x.device)
        for k in range(self.configs.order):
            modified_input = torch.tensor(invarient_x, device=invarient_x.device)

            modified_input[:, :k] = torch.matmul(pred_k_structures[:, :k],
                                                 invarient_x[:, :k].unsqueeze(dim=-1)).squeeze(dim=-1)
            coeff_k_with_weight = self.coeff_net(modified_input.view(-1, self.configs.order * self.configs.num_vars))

            coeff_k_with_weight = torch.reshape(coeff_k_with_weight,
                                                [-1, self.configs.num_vars, self.configs.num_vars, 2])

            k_lag_structure = F.gumbel_softmax(logits=coeff_k_with_weight, tau=1, hard=True)
            all_lag_structures[:, k] = k_lag_structure
            pred_k_structures[:, k] = k_lag_structure[..., 0]

        return pred_k_structures, all_lag_structures

    def calculate_kld(self, all_lag_structures: torch.tensor):
        posterior_dist = torch.distributions.Categorical(logits=all_lag_structures)
        prior_dist = torch.distributions.Categorical(probs=torch.ones_like(all_lag_structures) * 0.1)

        KLD = kl_divergence(posterior_dist, prior_dist).mean()

        return KLD

    def calculate_structure_loss(self, coeffs, all_lag_structures):
        subset_num = self.configs.seq_len // self.configs.subset_length
        length = (self.configs.subset_length - self.configs.order) * subset_num
        sparsity_penalty = 0.5 * torch.mean(torch.norm(coeffs, dim=(-1, -2), p=2)) + \
                           0.5 * torch.mean(torch.norm(coeffs, dim=(-1, -2), p=1))

        KLD = self.calculate_kld(all_lag_structures=all_lag_structures)
        coeffs = coeffs.reshape(-1, length, self.configs.order, self.configs.num_vars, self.configs.num_vars)
        diffs = coeffs[:, 1:] - coeffs[:, :-1]
        # batch_size:32,length=20,num_var:7  32*19*(49),norm
        norm1_sum = torch.mean(
            torch.norm(diffs.view(-1, (length - 1) * self.configs.order, self.configs.num_vars ** 2), p=1, dim=-1))
        norm2_sum = torch.mean(
            torch.norm(diffs.view(-1, (length - 1) * self.configs.order, self.configs.num_vars ** 2), p=2, dim=-1))

        other_loss = sparsity_penalty * self.configs.similar_weight + KLD + (
                norm1_sum + norm2_sum) * self.configs.sparse_weight
        return other_loss
