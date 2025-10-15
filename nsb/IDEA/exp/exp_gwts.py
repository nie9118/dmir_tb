import os

from matplotlib import pyplot as plt

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

from loss.New_GW_Loss import GWTS_LOSS
from loss.dilate_loss import Dilate_Loss
from loss.gw_cost_constraint import GW_CC
from loss.gw_zq import GW
from loss.new_gw_mse import New_GW_MSE
from utils.tools import EarlyStopping, adjust_learning_rate, create_path_gw, get_setting
from utils.metrics import metric_gw
import torch
import torch.nn as nn
from torch import optim
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_GWTS(Exp_Basic):
    def __init__(self, args):
        super(Exp_GWTS, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        self.path, self.logger = create_path_gw(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name):
        if loss_name == "mse":
            criterion = nn.MSELoss()
        elif loss_name == "GWW":
            criterion = GWTS_LOSS(self.args.beta, self.args.inner_N)  # 1,20

        elif loss_name == "dilate":
            criterion = Dilate_Loss(self.args.dilate_alpha, self.args.gamma)

        elif loss_name == 'new_gw':
            criterion = GW(self.args.lr, self.args.iteration, self.args.w_weight, self.args.gw_weight,
                           self.args.c_weight, move_size_list=self.args.move_sizes)

        elif loss_name == 'new_gw_mse':
            criterion = New_GW_MSE(self.args.lr, self.args.iteration, self.args.w_weight, self.args.gw_weight,
                                   self.args.move_sizes, 0)
        elif loss_name == 'gw_cc':
            criterion = GW_CC(self.args.lr, self.args.iteration_range, self.args.w_rate_range, self.args.gw_rate_range,
                              self.args.c_rate_range, move_size_list=self.args.move_sizes, type=self.args.gw_type,
                              adj_weight_type=self.args.adj_weight_type,
                              feature_cat_type=self.args.feature_cat_type, cost_type=self.args.cost_type)
        self.setting = get_setting(self.args)
        self.logger.info(self.setting)
        self.mse = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, epoch, rate):
        total_loss = []
        mse_loss_list = []
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
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                if self.args.loss == "GWW" or self.args.loss == "new_gw" or self.args.loss == "gw_cc":
                    loss = criterion(outputs, batch_y, rate)
                elif self.args.loss == "new_gw_mse":
                    loss = criterion(outputs, batch_y, epoch)
                else:
                    loss = criterion(outputs, batch_y)

                mse_loss = self.mse(outputs, batch_y)
                mse_loss_list.append(mse_loss.item())
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        avg_mse = np.average(mse_loss_list)
        self.model.train()
        return total_loss, avg_mse

    def train(self):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        train_steps = len(train_loader)
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        # model_param = torch.load('./model.pt')
        # self.model.load_state_dict(model_param)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            # if self.args.loss == "new_gw":
            #     criterion.iterations = iterations[epoch]
            #     criterion.learn_rate = lrs[epoch]
            iter_count = 0
            train_loss = []
            rate = epoch / self.args.train_epochs
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                loss = None
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
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
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:

                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if self.args.loss == "new_gw" or self.args.loss == "gw_cc":
                        # loss = criterion(outputs, batch_y, (epoch + 1) / self.args.train_epochs)

                        loss = criterion(outputs, batch_y, rate)
                    elif self.args.loss == "new_gw_mse":
                        loss = criterion(outputs, batch_y, epoch)
                    else:
                        loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss, val_avg_mse = self.vali(vali_data, vali_loader, criterion, epoch, rate)
            test_loss, test_avg_mse = self.vali(test_data, test_loader, criterion, epoch, rate)

            self.logger.info(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            self.logger.info(" Vali mse: {0:.7f} Test mse: {1:.7f}".format(val_avg_mse, test_avg_mse))
            self.early_stopping(vali_loss, self.model, self.path, is_saved=False)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args, self.logger)
            self.logger.info("")

    def test(self):
        test_data, test_loader = self._get_data(flag="test")

        preds = []
        trues = []
        inputs = []

        self.model.load_state_dict(self.early_stopping.best_model_params)
        # torch.save(self.early_stopping.best_model_params, "./model.pt")
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
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)

                preds.append(outputs)
                trues.append(batch_y)
                inputs.append(batch_x)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = np.concatenate(inputs, axis=0)

        mae, mse, rmse, mape, mspe, dtw, tdi = metric_gw(preds, trues)

        self.logger.info('rmse:{} mse:{}, mae:{},dtw:{},tdi:{}'.format(rmse, mse, mae, dtw, tdi))
        f = open(f"{self.path}/{self.args.loss}__{self.args.seq_len}_{self.args.pred_len}_{self.args.result_name}.txt",
                 'a')
        f.write(f'{self.setting}\n')
        f.write('rmse:{} ,mse:{}, mae:{},dtw:{},tdi:{}'.format(rmse, mse, mae, dtw, tdi))
        f.write('\n\n')
        f.close()
        
        if self.args.is_draw:
            print("start draw")
            for i in range(len(preds)):
                root_path = f"./draw/{self.args.model}/{self.args.root_path.rsplit('/')[-2]}_{self.args.data_path.replace('.csv', '')}/{self.args.loss}_{self.args.seq_len}_{self.args.pred_len}"
                os.makedirs(root_path, exist_ok=True)
                path = os.path.join(root_path, f'{i}.pdf')
                self.draw(inputs[i], preds[i], trues[i], path)

        return

    def draw(self, input, pred, target, path):
        input = np.squeeze(input)
        pred = np.squeeze(pred)
        target = np.squeeze(target)
        input_len = len(input)
        output_len = len(pred)
        plt.plot(range(0, input_len), input, label='input', linewidth=3)
        plt.plot(range(input_len - 1, input_len + output_len),
                 np.concatenate([input[input_len - 1:input_len], target]),
                 label='target', linewidth=3)
        plt.plot(range(input_len - 1, input_len + output_len), np.concatenate([input[input_len - 1:input_len], pred]),
                 label='prediction', linewidth=3)
        plt.legend()
        plt.savefig(path, bbox_inches='tight')
        plt.close()
