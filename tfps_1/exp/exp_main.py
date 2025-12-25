from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, PatchTST_MoE_cluster
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from sklearn.cluster import KMeans

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from thop import profile

from layers.Cluster import EDESC
from layers.InitializeD import Initialization_D
from layers.RevIN import RevIN

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _get_model(self):
        """Get the underlying model, handling DataParallel wrapping"""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def _get_model_attribute(self, attr_path):
        """Safely get nested model attributes, handling DataParallel wrapping"""
        base_model = self._get_model()
        attrs = attr_path.split('.')
        current = base_model
        for attr in attrs:
            current = getattr(current, attr)
        return current

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'PatchTST_MoE_cluster': PatchTST_MoE_cluster,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            # Detect AMD GPU environment (ROCm/HIP)
            is_amd_gpu = hasattr(torch.version, 'hip') and torch.version.hip is not None

            # Validate device availability
            available_devices = torch.cuda.device_count()
            if available_devices == 0:
                raise RuntimeError("No GPUs available. Please check your GPU setup.")

            # Validate requested device IDs
            for device_id in self.args.device_ids:
                if device_id >= available_devices:
                    raise RuntimeError(f"Requested GPU {device_id} not available. Only {available_devices} GPUs detected.")

            if is_amd_gpu:
                # For AMD GPUs using ROCm/HIP - PyTorch still uses 'cuda:' device naming
                print(f'Using AMD GPUs with ROCm/HIP, device IDs: {self.args.device_ids}')
                print(f'ROCm version: {torch.version.hip}')
                print(f'Available AMD GPUs: {available_devices}')
                primary_device = torch.device(f'cuda:{self.args.device_ids[0]}')
                model = model.to(primary_device)
                # Use DataParallel with AMD GPU device IDs
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
                print(f'Model successfully wrapped with DataParallel on AMD GPUs: {self.args.device_ids}')
            else:
                # For NVIDIA GPUs using CUDA
                print(f'Using NVIDIA GPUs with CUDA, device IDs: {self.args.device_ids}')
                if hasattr(torch.version, 'cuda'):
                    print(f'CUDA version: {torch.version.cuda}')
                print(f'Available NVIDIA GPUs: {available_devices}')
                primary_device = torch.device(f'cuda:{self.args.device_ids[0]}')
                model = model.to(primary_device)
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
                print(f'Model successfully wrapped with DataParallel on NVIDIA GPUs: {self.args.device_ids}')
        elif self.args.use_gpu:
            # Single GPU case - move to device
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_profile(self, model):
        _input=torch.randn(self.args.batch_size, self.args.seq_len, self.args.enc_in).to(self.device)
        macs, params = profile(model, inputs=(_input,))
        return macs, params

    def _refined_subspace_affinity(self, s):
        """Numerically stable refined subspace affinity.

        s: Tensor [B, K] with s>=0 and rows sum to 1 (ideally). We protect against zeros/NaNs.
        """
        eps = 1e-12
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s = torch.clamp(s, min=eps)
        col_sum = torch.sum(s, dim=0, keepdim=True)
        weight = s ** 2 / torch.clamp(col_sum, min=eps)
        row_sum = torch.sum(weight, dim=1, keepdim=True)
        weight = weight / torch.clamp(row_sum, min=eps)
        return weight

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        s_time, s_frequency, outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Update refined subspace affinity
                tmp_s_time = s_time.data
                s_tilde_time = self._refined_subspace_affinity(s=tmp_s_time)
                tmp_s_frequency = s_frequency.data
                s_tilde_frequency = self._refined_subspace_affinity(s=tmp_s_frequency)

                # Total loss function
                n_z = self.args.c_out * self.args.d_model
                T_dim = int(n_z / self.args.T_num_expert)
                F_dim = int(n_z / self.args.F_num_expert)

                # Get cluster loss using safe attribute access
                loss_cluster_time = self._get_model_attribute('model_time.cluster.total_loss')(
                    pred=s_time, target=s_tilde_time, dim=T_dim, n_clusters=self.args.T_num_expert, beta=self.args.beta)
                loss_cluster_frequency = self._get_model_attribute('model_frequency.cluster.total_loss')(
                    pred=s_frequency, target=s_tilde_frequency, dim=F_dim, n_clusters=self.args.F_num_expert, beta=self.args.beta)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true) + self.args.alpha * loss_cluster_time + self.args.gama * loss_cluster_frequency

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        print('Trainable parameters: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    # 在第一个iteration添加详细的输入检查
                    if i == 0 and epoch == 0:
                        print(f"\n=== First Iteration Debug Info ===")
                        print(f"batch_x shape: {batch_x.shape}, range: [{batch_x.min().item():.6f}, {batch_x.max().item():.6f}]")
                        print(f"batch_x contains NaN: {torch.isnan(batch_x).any().item()}")
                        print(f"batch_x contains Inf: {torch.isinf(batch_x).any().item()}")

                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        s_time, s_frequency, outputs = self.model(batch_x)

                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            print(f"\nDEBUG: Model output contains NaN/Inf at iteration {i+1}")
                            print(
                                f"  batch_x stats: min={batch_x.min().item():.6f}, max={batch_x.max().item():.6f}, mean={batch_x.mean().item():.6f}")
                            print(
                                f"  s_time stats: min={s_time.min().item() if torch.isfinite(s_time).any() else 'non_finite'}, max={s_time.max().item() if torch.isfinite(s_time).any() else 'non_finite'}")
                            print(
                                f"  s_frequency stats: min={s_frequency.min().item() if torch.isfinite(s_frequency).any() else 'non_finite'}, max={s_frequency.max().item() if torch.isfinite(s_frequency).any() else 'non_finite'}")
                    else:
                        # BUGFIX: this branch must call the standard encoder-decoder signature.
                        # Previously it incorrectly passed batch_y and could corrupt outputs.
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # For non-PatchTST models we don't have cluster losses.
                        s_time, s_frequency = None, None

                    # If PatchTST_MoE_cluster path, compute cluster losses; otherwise keep them 0.
                    loss_cluster_time = torch.tensor(0.0, device=self.device)
                    loss_cluster_frequency = torch.tensor(0.0, device=self.device)
                    if s_time is not None and s_frequency is not None:
                        tmp_s_time = s_time.data
                        s_tilde_time = self._refined_subspace_affinity(s=tmp_s_time)
                        tmp_s_frequency = s_frequency.data
                        s_tilde_frequency = self._refined_subspace_affinity(s=tmp_s_frequency)

                        n_z = self.args.c_out * self.args.d_model
                        T_dim = int(n_z / self.args.T_num_expert)
                        F_dim = int(n_z / self.args.F_num_expert)

                        loss_cluster_time = self._get_model_attribute('model_time.cluster.total_loss')(
                            pred=s_time, target=s_tilde_time, dim=T_dim, n_clusters=self.args.T_num_expert,
                            beta=self.args.beta)
                        loss_cluster_frequency = self._get_model_attribute('model_frequency.cluster.total_loss')(
                            pred=s_frequency, target=s_tilde_frequency, dim=F_dim,
                            n_clusters=self.args.F_num_expert, beta=self.args.beta)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss_fore = criterion(outputs, batch_y)
                    loss = loss_fore + self.args.alpha * loss_cluster_time + self.args.gama * loss_cluster_frequency

                    if (not torch.isfinite(loss)):
                        print(f"Warning: NaN or Inf loss detected at iteration {i+1}")
                        try:
                            print(
                                f"loss_fore: {loss_fore.item()}, loss_cluster_time: {loss_cluster_time.item()}, loss_cluster_frequency: {loss_cluster_frequency.item()}")
                            print(f"outputs range: [{outputs.min().item()}, {outputs.max().item()}]")
                            print(f"batch_y range: [{batch_y.min().item()}, {batch_y.max().item()}]")
                        except Exception:
                            pass
                        # Robustness: clear any bad grads and skip this batch
                        model_optim.zero_grad(set_to_none=True)
                        continue

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    # If any grad becomes non-finite, skip the step to avoid corrupting weights.
                    grads_finite = True
                    for p in self.model.parameters():
                        if p.grad is not None and (not torch.isfinite(p.grad).all()):
                            grads_finite = False
                            break
                    if not grads_finite:
                        print(f"Warning: Non-finite gradients at iteration {i+1}, skipping optimizer step")
                        model_optim.zero_grad(set_to_none=True)
                        continue

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # total_params = sum(p.numel() for p in self.model.parameters())

        # print("模型总参数数量:", total_params)

        preds = []
        trues = []
        clusters_time = []
        clusters_frequency = []
        inputx = []
        inference_time = 0  # 初始化 inference_time
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                start_time = time.time()  # 计时开始
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            s_time, s_frequency, outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                inference_time += time.time() - start_time  # 计算推理时间
                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if i == 0:
                    print("\n...")

                    # 1. 处理 Input X (输入序列)
                    # batch_x 还在 GPU 上，需要转 numpy
                    # 原始 shape: [Batch, Seq_Len, Features] -> 取 [0] -> [Seq_Len, Features]
                    sample_input_x = batch_x.detach().cpu().numpy()[0, :, :]

                    # 2. 处理 Truth Y (真实标签)
                    # batch_y 已经是 numpy
                    # 原始 shape: [Batch, Pred_Len, Features] -> 取 [0] -> [Pred_Len, Features]
                    sample_truth_y = batch_y[0, :, :]

                    # 3. 处理 Output (模型预测)
                    # outputs 已经是 numpy
                    # 原始 shape: [Batch, Pred_Len, Features] -> 取 [0] -> [Pred_Len, Features]
                    sample_pred_out = outputs[0, :, :]

                    # 定义保存路径
                    debug_dir = os.path.join(folder_path, 'debug_data')
                    if not os.path.exists(debug_dir):
                        os.makedirs(debug_dir)

                    # 保存为 CSV (不保存行索引，保留列结构)
                    # 如果你知道特征的名字，可以在 pd.DataFrame 中加 columns=['feat1', 'feat2'...]
                    pd.DataFrame(sample_input_x).to_csv(
                        os.path.join(debug_dir, 'sample_0_input_x.csv'), header=False, index=False
                    )
                    pd.DataFrame(sample_truth_y).to_csv(
                        os.path.join(debug_dir, 'sample_0_truth_y.csv'), header=False, index=False
                    )
                    pd.DataFrame(sample_pred_out).to_csv(
                        os.path.join(debug_dir, 'sample_0_pred_out.csv'), header=False, index=False
                    )
                    print(f"CSV: {debug_dir}")
                    print(
                        f"Shapes -> Input: {sample_input_x.shape}, True: {sample_truth_y.shape}, Pred: {sample_pred_out.shape}")

                cluster_time = s_time.detach().cpu().numpy()
                cluster_frequency = s_frequency.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                cluster_time = cluster_time
                cluster_frequency = cluster_frequency

                preds.append(pred)
                trues.append(true)
                clusters_time.append(cluster_time)
                clusters_frequency.append(cluster_frequency)

                inputx.append(batch_x.detach().cpu().numpy())
                # if i % 20 == 0:
                    # input = batch_x.detach().cpu().numpy()
                    # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        clusters_time = np.array(clusters_time)
        clusters_frequency = np.array(clusters_frequency)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # 将总体推理时间除以推理次数，得到平均推理时间
        total_samples = len(test_loader)
        if total_samples > 0:
            inference_time /= total_samples

        # result save
        try:
            folder_path = os.path.join('./results', setting)
            os.makedirs(folder_path, exist_ok=True)

            # 安全处理文件名
            data_name = self.args.data_path.split('.')[0] if '.' in self.args.data_path else self.args.data_path
            result_file = os.path.join('.', f'result_{data_name}_{self.args.seq_len}.txt')

            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            print('mse:{}, mae:{}, rse:{}, Average Inference Time:{}, total_params:{}'.format(
                mse, mae, rse, inference_time, total_params))
            
            # 使用上下文管理器安全写入文件
            with open(result_file, 'a', encoding='utf-8') as f:
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}, rse:{}, Average Inference Time:{}, total_params:{}\n'.format(
                    mse, mae, rse, inference_time, total_params))
                f.write('\n')
            print(f'Results saved to {result_file}')

            # 保存numpy数组,添加异常处理
            try:
                # np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, rse, corr]))
                np.save(os.path.join(folder_path, 'cluster_time_result.npy'), clusters_time)
                np.save(os.path.join(folder_path, 'cluster_frequency_result.npy'), clusters_frequency)
                np.save(os.path.join(folder_path, 'pred.npy'), preds)
                # np.save(os.path.join(folder_path, 'true.npy'), trues)
                # np.save(os.path.join(folder_path, 'x.npy'), inputx)
                print(f'Numpy arrays saved to {folder_path}')
            except Exception as e:
                print(f'Warning: Failed to save numpy arrays: {str(e)}')
                
        except Exception as e:
            print(f'Error saving results: {str(e)}')
            import traceback
            traceback.print_exc()
        
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
