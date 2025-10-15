import argparse

import time
import optuna
import torch
import os
from exp.exp_nsts import Exp_NSTS

from exp.exp_nsts_with_pre import Exp_NSTS_Pre
from utils.tools import setSeed

def objective(trial):
    parser = argparse.ArgumentParser()

    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)

    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--model', type=str, default='NSTS',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/illness/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='national_illness.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./nsts', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--h_dim', type=int, default=20, help='inverse output data')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimizationF
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--save_time', action='store_true', default=False, help='whether to output attention in ecoder')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--seed', type=int, default=2022, help='number of hidden layers in projector')
    parser.add_argument('--WITRAN_deal', type=str, default='None',
                        help='WITRAN deal data type, options:[None, standard]')
    parser.add_argument('--WITRAN_grid_cols', type=int, default=24,
                        help='Numbers of data grid cols for WITRAN')

    # NSTS
    parser.add_argument('--zc_dim', type=int, default=7, help='num of encoder layers')
    parser.add_argument('--zd_dim', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--zd_kl_weight', type=float, default=1, help='num of encoder layers')
    parser.add_argument('--zc_kl_weight', type=float, default=0.001, help='num of encoder layers')
    parser.add_argument('--hmm_weight', type=float, default=0.001, help='num of encoder layers')
    # parser.add_argument('--rec_weight', type=float, default=0.5, help='latent dimension of koopman embedding')
    parser.add_argument('--n_class', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--No_prior', action='store_true', default=False, help='num of encoder layers')
    parser.add_argument('--lags', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--embedding_dim', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--is_bn', action='store_true', default=False, help='num of encoder layers')
    parser.add_argument('--dynamic_dim', type=int, default=128, help='latent dimension of koopman embedding')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of en/decoder')
    parser.add_argument('--hidden_layers', type=int, default=2, help='number of hidden layers of en/decoder')

    parser.add_argument('--pre_epoches', type=int, default=1, help='train epochs')

    # idol
    parser.add_argument('--z_dim', type=int, default=8, help='num of encoder layers')
    parser.add_argument('--kl_weight', type=float, default=0.001, help='num of encoder layers')
    parser.add_argument('--sparsity_weight', type=float, default=0.001, help='num of encoder layers')
    parser.add_argument('--rec_weight', type=float, default=0.5, help='num of encoder layers')

    parser.add_argument('--emb_dim', type=int, default=48, help='dimension of model')

    parser.add_argument('--patch_size', type=int, default=24, help='size of patches')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    #设置optuna超参
    args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    args.batch_size = trial.suggest_categorical('batch_size', [16,32,48,64])


    #GPU设置
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.No_prior:

        Exp = Exp_NSTS
    else:

        Exp = Exp_NSTS_Pre

    setSeed(args.seed)
    exp = Exp(args)  # set experiments
    start_time = time.time()

    print(f"\n--- [Trial {trial.number}] 开始训练 ---")
    param_str = ", ".join([f"{k}={v}" for k, v in trial.params.items()])
    print(param_str)
    print(f"dataset: {args.data}")
    print(f"seq_len: {args.seq_len}")
    print(f"pred_len: {args.pred_len}")

    exp.train()
    object,test_mae, test_mse = exp.test()

    trial.set_user_attr("test_mae", test_mae)
    trial.set_user_attr("test_mse", test_mse)

    torch.cuda.empty_cache()
    return object

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='getting file name')
    parser.add_argument('--model', type=str, default='NSTS',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset name')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    args, unknown = parser.parse_known_args()

    study = optuna.create_study(direction='minimize')


#   n_trials控制搜索次数
    study.optimize(objective, n_trials=1)

    print("\n\n--- 优化完成 ---")
    print("完成的试验次数: ", len(study.trials))
    print("最佳试验:")
    best_trial = study.best_trial

    print(f"  > 最佳测试损失 (MAE+MSE): {best_trial.value:.9f}")
    print(f"  > 对应的测试 MAE: {best_trial.user_attrs['test_mae']:.9f}")
    print(f"  > 对应的测试 MSE: {best_trial.user_attrs['test_mse']:.9f}")
    print("  > 最佳超参数 (Params): ")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")

    output_dir = 'optuna'
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(args.data_path)
    filename_without_ext = os.path.splitext(base_filename)[0]

    filename = f"{filename_without_ext}_model_{args.model}_seqlen_{args.seq_len}_predlen_{args.pred_len}_results.txt"
    file_path = os.path.join(output_dir, filename)

    print(f"\n准备将最佳结果写入到: {file_path}")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(
            f"--- Optuna Results for Dataset: {args.data}, Seq_len: {args.seq_len}, Pred Len: {args.pred_len} ---\n\n")
        f.write(f"Total trials completed: {len(study.trials)}\n\n")
        f.write("--- Best Trial ---\n")
        f.write(f"Objective Value (mae+mse): {best_trial.value:.9f}\n")
        f.write(f"Corresponding Test MAE: {best_trial.user_attrs['test_mae']:.9f}\n")
        f.write(f"Corresponding Test MSE: {best_trial.user_attrs['test_mse']:.9f}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  - {key}: {value}\n")

    print(f"最佳结果已成功写入！")

    print(f"\n最佳结果已成功写入到: {file_path}")
    end_time = time.time()
    print(f"总耗时：{end_time - start_time:.2f}s")