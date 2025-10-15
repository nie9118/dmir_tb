import time

import optuna
import torch
import random
import numpy as np
import argparse
import os

# 导入你的实验类
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast


def objective(trial):
    """
    Optuna 的目标函数。每一次调用都会使用一组新的超参数来训练模型。
    """
    # ---- 固定随机种子 ----
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimeBridge')

    # ablation control flags
    parser.add_argument('--revin', action='store_false', help='non-stationary for short-term', default=True)
    parser.add_argument('--alpha', type=float, default=0.2, help='weight of time-frequency MAE loss')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--attn_dropout', type=float, default=0.15, help='dropout')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

    # basic config
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='ETTh1_sl720_TimeBridge_optuna')
    parser.add_argument('--model', type=str, default='TimeBridge')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                        help='start token length')  # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--ia_layers', type=int, default=3, help='num of integrated attention layers')
    parser.add_argument('--pd_layers', type=int, default=1, help='num of patch downsampled layers')
    parser.add_argument('--ca_layers', type=int, default=0, help='num of cointegrated attention layers')

    parser.add_argument('--stable_len', type=int, default=6, help='length of moving average in patch norm')
    parser.add_argument('--num_p', type=int, default=None, help='num of down sampled patches')

    parser.add_argument('--period', type=int, default=24, help='length of patches')

    parser.add_argument('--enc_in', type=int, default=7, help='channel_decoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')

    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--embedding_epochs', type=int, default=5, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--pct_start', type=float, default=0.2, help='optimizer learning rate')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--embedding_lr', type=float, default=0.0005, help='optimizer learning rate of embedding')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2', help='device ids of multile gpus')

    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    parser.add_argument('--random', type=bool,default=True)

    # ---- 2. 定义超参数搜索空间 ----
    # Optuna 将从这里动态地建议超参数，覆盖默认值
    args = parser.parse_args()  # 使用空列表来避免解析命令行

    args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    args.batch_size = trial.suggest_categorical('batch_size', [16,32,48,64])

    # args.ca_layers = trial.suggest_categorical('ca_layers', [0,1,2,3])
    # args.pd_layers = 1
    # args.ia_layers = trial.suggest_categorical('ia_layers', [1,2,3])
    # if args.ca_layers >= args.ia_layers:
    #     raise optuna.exceptions.TrialPruned()
    #
    # possible_n_heads = [h for h in [4, 8, 16, 32] if args.d_model % h == 0]
    # if not possible_n_heads:  # 如果没有可用的 n_heads，则跳过此次试验
    #     raise optuna.exceptions.TrialPruned()
    # args.n_heads = trial.suggest_categorical('n_heads', possible_n_heads)
    # args.num_p = trial.suggest_categorical('num_p', [4,8,12])
    # # d_ff 通常是 d_model 的倍数
    # args.d_ff = trial.suggest_categorical('d_ff_multiplier', [1, 2, 4]) * args.d_model

    # 打印本次试验的参数
    print(f"\n--- [Trial {trial.number}] 参数 ---")
    param_str = ", ".join([f"{k}={v}" for k, v in trial.params.items()])
    print(param_str)

    # ---- 3. 运行实验 ----
    # 设置 GPU
    if args.use_gpu:
        torch.cuda.set_device(args.gpu)

    # 实例化实验
    exp = Exp_Long_Term_Forecast(args)

    # 构造 setting 字符串
    setting = '{}_{}_trial{}'.format(
        args.model_id,
        args.data,
        trial.number
    )

    print(f"\n--- [Trial {trial.number}] 开始训练 ---")
    print(f"dataset: {args.data}")
    print(f"seq_len: {args.seq_len}")
    print(f"pred_len: {args.pred_len}")
    # 运行训练并获取最佳验证损失 (这依赖于第一步的修改)
    exp.train(setting,trial)
    # 运行测试并获取测试集的 mae 和 mse
    object ,test_mae, test_mse = exp.test(setting)

    # ---- 将附加指标存入 user_attrs ----
    trial.set_user_attr("test_mae", test_mae)
    trial.set_user_attr("test_mse", test_mse)



    # 清理 GPU 缓存
    torch.cuda.empty_cache()

    # ---- 4. 返回评估指标 ----
    return object


# ---- 5. 创建 Study 并开始优化 ----
if __name__ == '__main__':
    # 'minimize' 表示我们的目标是让 objective 函数的返回值（验证损失）最小化
    start_time = time.time()
    parser = argparse.ArgumentParser(description='getting file name')
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset name')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    args, unknown = parser.parse_known_args()

    study = optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner())

    # 'n_trials' 是你想要尝试的超参数组合的总次数
    # 从一个较小的数字开始，比如 20，然后再增加
    study.optimize(objective, n_trials=40)

    # ---- 6. 输出优化结果 ----
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

    # ---- 7. 将最佳结果写入文件 ----
    output_dir = 'optuna'
    os.makedirs(output_dir, exist_ok=True)  # 确保文件夹存在
    # 从 data_path 中提取基本文件名，以避免路径问题
    # 例如, 从 './dataset/ETTh1.csv' 提取出 'ETTh1'
    base_filename = os.path.basename(args.data_path)  # 获取 'ETTh1.csv'
    filename_without_ext = os.path.splitext(base_filename)[0] # 获取 'ETTh1'

    # 根据 dataset 和 pred_len 动态生成文件名
    filename = f"{filename_without_ext}_seqlen_{args.seq_len}_predlen_{args.pred_len}_results_v3.txt"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(
            f"--- Optuna Results for Dataset: {args.data}, Seq_len: {args.seq_len}, Pred Len: {args.pred_len} ---\n\n")
        f.write(f"Total trials completed: {len(study.trials)}\n\n")
        f.write("--- Best Trial ---\n")
        f.write(f"Objective Value (mae+mse): {best_trial.value:.7f}\n")
        f.write(f"Corresponding Test MAE: {best_trial.user_attrs['test_mae']:.7f}\n")
        f.write(f"Corresponding Test MSE: {best_trial.user_attrs['test_mse']:.7f}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  - {key}: {value}\n")

    print(f"最佳结果已成功写入！")

    print(f"\n最佳结果已成功写入到: {file_path}")
    end_time = time.time()
    print(f"总耗时：{end_time - start_time:.2f}s")