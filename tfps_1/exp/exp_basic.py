import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()
        # Don't move model to device here if using multi-GPU, as it will be handled in _build_model

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # Detect AMD GPU environment (ROCm/HIP)
            is_amd_gpu = hasattr(torch.version, 'hip') and torch.version.hip is not None

            if is_amd_gpu:
                # For AMD GPUs using ROCm/HIP - PyTorch still uses 'cuda:' device naming
                os.environ["HIP_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use AMD GPU with ROCm/HIP: cuda:{}'.format(self.args.gpu))
            else:
                # For NVIDIA GPUs using CUDA
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use NVIDIA GPU with CUDA: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
