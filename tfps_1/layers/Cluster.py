import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from layers.Constraint import D_constraint1, D_constraint2


class EDESC(nn.Module):

    def __init__(self,
                 d_model,
                 n_clusters,
                 eta,
                 c_out,
                 bs,
                 patch_len,
                 stride):
        super(EDESC, self).__init__()
        self.n_clusters = n_clusters
        self.eta = eta
        self.c_out = c_out
        self.bs = bs
        self.patch_len = patch_len
        self.stride = stride

        # numerical stability
        self.eps = 1e-12

        # Subspace bases proxy
        n_z = c_out * d_model
        self.d = int(n_z / n_clusters)

        D_init = torch.empty(n_clusters * self.d, n_clusters * self.d)
        nn.init.orthogonal_(D_init)
        self.D = Parameter(D_init)

    def reverse_unfold(self, z, original_length, stride):
        # z: [bs x patch_num x nvars x patch_len]
        bs, patch_num, nvars, patch_len = z.size()
        output = torch.zeros((bs, nvars, original_length), device=z.device)
        patch_counts = torch.zeros((bs, nvars, original_length), device=z.device)

        for i in range(patch_num):
            start = i * stride
            end = start + patch_len
            if end > original_length:
                end = original_length

            current_patch_len = end - start

            output[:, :, start:end] += z[:, i, :, :current_patch_len]
            patch_counts[:, :, start:end] += 1

        # avoid divide-by-zero
        output = output / torch.clamp(patch_counts, min=1.0)
        output = torch.reshape(output, (output.shape[0], output.shape[2], output.shape[1]))
        return output   # [bs, c_out, context_window]

    def forward(self, z):  # z: [bs * patch_num_out x nvars * d_model]
        s = None

        # Calculate subspace affinity
        for i in range(self.n_clusters):
            si = torch.sum(torch.pow(torch.mm(z, self.D[:, i * self.d:(i + 1) * self.d]), 2), 1, keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s, si), 1)

        s = (s + self.eta * self.d) / ((self.eta + 1) * self.d)

        # stable normalization (Eq 13)
        denom = torch.sum(s, dim=1, keepdim=True)
        s = s / torch.clamp(denom, min=self.eps)
        s = torch.clamp(s, min=self.eps, max=1.0)
        s = s / torch.sum(s, dim=1, keepdim=True)
        return s, z

    def total_loss(self, pred, target, dim, n_clusters, beta):
        # Ensure valid probability simplices for KL
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        pred = pred / torch.sum(pred, dim=-1, keepdim=True)

        target = torch.clamp(target, min=self.eps, max=1.0)
        target = target / torch.sum(target, dim=-1, keepdim=True)

        # Subspace clustering loss  Eq 15
        # Use stable KL: KL(target || pred) = sum target * (log(target) - log(pred))
        kl_loss = F.kl_div(torch.log(pred), target, reduction='batchmean', log_target=False)

        # Constraints   Eq 12
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()
        # Call forward explicitly to avoid confusion with IDE inspection
        loss_d1 = d_cons1.forward(self.D)
        loss_d2 = d_cons2.forward(self.D, dim, n_clusters)

        total_loss = beta * kl_loss + loss_d1 + loss_d2
        return total_loss
