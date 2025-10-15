import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.nn import init
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, configs):

        super(Model, self).__init__()
        self.lambdas = configs.lambdas
        self.structure_similarity_restric = configs.structure_similarity_restric
        self.src_pred_list = nn.ModuleList()
        self.coeff_nets = nn.ModuleList()
        self.num_vars = configs.dec_in
        self.hidden_layer_size = configs.hidden_layer_size
        self.num_hidden_layer = configs.num_hidden_layer
        self.order = configs.order
        self.configs = configs

        for k in range(self.order):
            modules = [nn.Sequential(nn.Linear(self.num_vars * self.order, self.hidden_layer_size))]
            if self.num_hidden_layer > 1:
                for j in range(self.num_hidden_layer - 1):
                    modules.extend(nn.Sequential(nn.Linear(self.hidden_layer_size, self.hidden_layer_size)))
            modules.extend(nn.Sequential(
                nn.BatchNorm1d(self.hidden_layer_size),
                nn.Linear(self.hidden_layer_size, self.num_vars * self.num_vars * 2)))
            self.coeff_nets.append(nn.Sequential(*modules))

        #  Decoder
        self.lag_pred_list = nn.ModuleList()

        for k in range(self.order):
            self.lag_pred_list.append(nn.ModuleList())
            for j in range(self.num_vars):
                self.lag_pred_list[k].append(nn.Sequential(nn.Linear(self.num_vars, 2 * self.num_vars),
                                                           nn.ReLU(),
                                                           nn.Linear(2 * self.num_vars, 1)
                                                           ))
                init.normal_(self.lag_pred_list[k][j][0].weight, mean=0.0, std=0.01)
                init.normal_(self.lag_pred_list[k][j][-1].weight, mean=0.0, std=0.01)

    def forward(self, inputs, hard=True):
        batch_size = inputs.shape[0]
        pred_k_structure_list = list()
        coeffs = torch.zeros(size=[batch_size, self.order, self.num_vars, self.num_vars]).to(inputs.device)
        all_lag_structures = torch.zeros([batch_size, self.order, self.num_vars, self.num_vars, 2]).to(
            inputs.device)
        preds = torch.zeros([batch_size, self.num_vars]).to(inputs.device)
        for k in range(self.order):
            modified_input_list = list()
            for i in range(self.order):
                if i < len(pred_k_structure_list):
                    modified_input = torch.matmul(pred_k_structure_list[i],
                                                  inputs[:, i, :].unsqueeze(-1)).squeeze(
                        dim=-1)
                else:
                    modified_input = inputs[:, i, :]

                modified_input_list.append(modified_input)

            total_modified_input = torch.cat(modified_input_list, dim=-1)

            coeff_net_k = self.coeff_nets[k]
            coeff_k_with_weight = coeff_net_k(total_modified_input)

            coeff_k_with_weight = torch.reshape(coeff_k_with_weight, [batch_size, self.num_vars, self.num_vars, 2])

            k_lag_structure = F.gumbel_softmax(logits=coeff_k_with_weight,tau=1, hard=hard)
            all_lag_structures[:, k, :, :, :] = k_lag_structure
            coeffs[:, k, :, :] = k_lag_structure[:, :, :, 0]
            coeff_k_with_weight=coeff_k_with_weight[:,:,:,0]
            pred_k_structure_list.append(coeff_k_with_weight)
            lag_pred_layer_list = self.lag_pred_list[k]
            mask_input =  coeff_k_with_weight* inputs[:, k, :].unsqueeze(1).repeat(
                [1, self.num_vars, 1])
            for var_idx in range(self.num_vars):
                lag_pred = lag_pred_layer_list[var_idx](mask_input[:, var_idx, :]).squeeze()
                preds[:, var_idx] += lag_pred
        other_loss = self.calculate_other_loss(coeffs, all_lag_structures)
        return preds, other_loss

    def getStruct_from_encoder(self, X):
        pred_k_structure_list = list()
        coeff_list = []
        for k in range(self.order):
            modified_input_list = list()
            for i in range(self.order):
                if i < len(pred_k_structure_list):
                    modified_input = torch.matmul(pred_k_structure_list[i],
                                                  X[:, i, :].unsqueeze(-1)).squeeze(
                        dim=-1)
                else:
                    modified_input = X[:, i, :]

                modified_input_list.append(modified_input)

            total_modified_input = torch.cat(modified_input_list, dim=-1)

            coeff_net_k = self.coeff_nets[k]
            coeff_k_with_weight = coeff_net_k(total_modified_input)
            coeff_k_with_weight = torch.reshape(coeff_k_with_weight, [X.shape[0], self.num_vars, self.num_vars, 2])[:,:,:,0]
            coeff_list.append(coeff_k_with_weight)
        coeff_list=torch.stack(coeff_list,dim=1)
        return coeff_list

    def getPred_from_decoder(self, X, A):
        batch_size = X.shape[0]
        preds = torch.zeros([batch_size, self.num_vars]).to(X.device)
        for k in range(self.order):
            lag_pred_layer_list = self.lag_pred_list[k]
            mask_input = A[:,k] * X[:, k, :].unsqueeze(1).repeat(
                [1, self.num_vars, 1])
            for var_idx in range(self.num_vars):
                lag_pred = lag_pred_layer_list[var_idx](mask_input[:, var_idx, :]).squeeze()
                preds[:, var_idx] += lag_pred
        return preds

    def calculate_kld(self, all_lag_structures: torch.tensor):
        posterior_dist = torch.distributions.Categorical(logits=all_lag_structures)
        prior_dist = torch.distributions.Categorical(probs=torch.ones_like(all_lag_structures) * 0.1)

        KLD = kl_divergence(posterior_dist, prior_dist).mean()

        return KLD

    def get_parameter_num(self):
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_num

    def calculate_other_loss(self, coeffs, all_lag_structures):
        length = self.configs.legnth_pre_sample
        sparsity_penalty = 0.5 * torch.mean(torch.norm(coeffs, dim=(-1, -2), p=2)) + \
                           0.5 * torch.mean(torch.norm(coeffs, dim=(-1, -2), p=1))

        KLD = self.calculate_kld(all_lag_structures=all_lag_structures)
        coeffs = coeffs.reshape(-1, length * self.configs.order, coeffs.shape[-2],
                                coeffs.shape[-1])
        diffs = coeffs[:, 1:] - coeffs[:, :-1]
        # batch_size:32,length=20,num_var:7  32*19*(49),norm
        norm1_sum = torch.mean(
            torch.norm(diffs.view(-1, length * self.configs.order - 1, self.num_vars ** 2), p=1, dim=-1))
        norm2_sum = torch.mean(
            torch.norm(diffs.view(-1, length * self.configs.order - 1, self.num_vars ** 2), p=2, dim=-1))

        other_loss = self.lambdas * sparsity_penalty + KLD + (norm1_sum + norm2_sum) * self.structure_similarity_restric
        return other_loss



