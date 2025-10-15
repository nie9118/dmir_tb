import os
import time
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, create_path
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import warnings
import numpy as np
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Type, Union

warnings.filterwarnings('ignore')

class Exp_NSTS(Exp_Basic):
    def __init__(self, args):
        super(Exp_NSTS, self).__init__(args)

    def _parse_type(self, str_or_type: Union[Type, str]) -> Type:
        if isinstance(str_or_type, str):
            return eval(str_or_type)
        elif isinstance(str_or_type, type):
            return str_or_type
        else:
            raise RuntimeError(f"{str_or_type} should be string or type")

    def _build_model(self):
        if self.args.model == 'FAN':
            n_model = self.model_dict['FAN_RevIN'].Model(self.args.enc_in)
            f_model = self.model_dict['FAN_DLinear'].Model(self.args)
            model = self.model_dict[self.args.model].Model(f_model, n_model).float()
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        self.path, self.logger = create_path(self.args)
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

    def vali(self, vali_loader, criterion, epoch):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                other_loss = 0
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            if self.args.model == "NSTS":
                                outputs, other_loss = self.model(batch_x, batch_y)
                            elif self.args.model == "MICN_idol":
                                outputs, other_loss = self.model(batch_x)
                            elif self.args.model == "NSTS1":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            elif self.args.model == "NSTS2":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            elif self.args.model == "NSTS3":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            elif self.args.model == "NSTS_Patch":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            elif self.args.model == "NSTS_iTransformer":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        if self.args.model == "NSTS":
                            outputs, other_loss = self.model(batch_x, batch_y)
                        elif self.args.model == "MICN_idol":
                            outputs, other_loss = self.model(batch_x)
                        elif self.args.model == "NSTS1":
                            outputs, other_loss = self.model(batch_x, batch_y)
                        elif self.args.model == "NSTS2":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        elif self.args.model == "NSTS3":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        elif self.args.model == "NSTS_Patch":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        elif self.args.model == "NSTS_iTransformer":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y) + other_loss
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        train_steps = len(train_loader)
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            # start_time = time.time()
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
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # encoder - decoder
                loss = None
                other_loss = 0
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            if self.args.model == "NSTS":
                                outputs, other_loss = self.model(batch_x, batch_y)
                            elif self.args.model == "MICN_idol":
                                outputs, other_loss = self.model(batch_x)
                            elif self.args.model == "NSTS1":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            elif self.args.model == "NSTS2":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            elif self.args.model == "NSTS3":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            elif self.args.model == "NSTS_Patch":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            elif self.args.model == "NSTS_iTransformer":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        if self.args.model == "NSTS":
                            outputs, other_loss = self.model(batch_x, batch_y)
                        elif self.args.model == "MICN_idol":
                            outputs, other_loss = self.model(batch_x)
                        elif self.args.model == "NSTS1":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        elif self.args.model == "NSTS2":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        elif self.args.model == "NSTS3":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        elif self.args.model == "NSTS_Patch":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        elif self.args.model == "NSTS_iTransformer":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y) + other_loss
                    train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion, epoch)
            test_loss = self.vali(test_loader, criterion, epoch)

            self.logger.info(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
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
                            if self.args.model == "NSTS":
                                outputs, other_loss = self.model(batch_x, batch_y)
                            elif self.args.model == "MICN_idol":
                                outputs, other_loss = self.model(batch_x)
                            elif self.args.model == "NSTS1":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            elif self.args.model == "NSTS2":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            elif self.args.model == "NSTS3":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            elif self.args.model == "NSTS_Patch":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            elif self.args.model == "NSTS_iTransformer":
                                outputs, other_loss = self.model(batch_x, batch_y)  
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        if self.args.model == "NSTS":
                            outputs, other_loss = self.model(batch_x, batch_y)
                        elif self.args.model == "MICN_idol":
                            outputs, other_loss = self.model(batch_x)
                        elif self.args.model == "NSTS1":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        elif self.args.model == "NSTS2":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        elif self.args.model == "NSTS3":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        elif self.args.model == "NSTS_Patch":
                            outputs, other_loss = self.model(batch_x, batch_y)  
                        elif self.args.model == "NSTS_iTransformer":
                            outputs, other_loss = self.model(batch_x, batch_y)  
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

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.logger.info('rmse:{} mse:{}, mae:{}'.format(rmse, mse, mae))
        f = open(f"{self.path}/{self.args.seq_len}_{self.args.pred_len}.txt", 'a')
        f.write(f"{str(self.args)}\n")
        f.write('rmse:{} ,mse:{}, mae:{}\n'.format(rmse, mse, mae))
        f.write('\n\n')
        f.close()

