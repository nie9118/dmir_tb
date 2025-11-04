import subprocess
import os
from itertools import product

# 设置环境变量（指定GPU）
os.environ["HIP_VISIBLE_DEVICES"] = "3"
os.environ["MIOPEN_DISABLE_CACHE"] = "1"
os.environ["MIOPEN_SYSTEM_DB_PATH"] = ""

# 配置基础参数
model_name = "TimeBridge"
data_name = "ETTm2"
root='./data' # 数据集根路径
data_path = 'ETT-small' # 可选[ETT-small，electricity，exchange_rate，illness，traffic，weather]
seq_len=96
pred_len=336 #36 48 60
alpha=0.395820936

enc_in=7

# 定义要搜索的参数网格
batch_sizes = [48]
learning_rates = [0.000160808]
ca_layers = [1]  # 长期
pd_layers = [1]
ia_layers = [2]  # 短期
seed=list(range(2000,2100))

# 生成所有参数组合
param_combinations = product(batch_sizes, learning_rates,ca_layers,pd_layers,ia_layers,seed)

# 遍历每个参数组合并执行命令
for batch_size,lr,ca_layers,pd_layers,ia_layers,seed in param_combinations:
    print(f"\n===== 开始执行参数组合: batch_size={batch_size}, learning_rate={lr}=====")

    # 构建命令列表
    command = [
        "python", "run.py",
        "--is_training", "1",
        "--root_path",f"{root}/{data_path}/",
        "--data_path",f"{data_name}.csv",
        "--model_id",f"{data_name}'_'{seq_len}'_'{pred_len}",
        "--model",f"{model_name}",
        "--data",f"{data_name}",
        "--features","M",
        "--seq_len",f"{seq_len}",
        "--label_len","48",
        "--pred_len",f"{pred_len}",
        "--enc_in",f"{enc_in}",
        "--ca_layers", str(ca_layers),
        "--pd_layers", str(pd_layers),
        "--ia_layers", str(ia_layers),
        "--des", "Exp",
        "--n_heads", "16",
        "--d_model", "64",
        "--d_ff", "128",
        "--lradj", 'TST',
        "--period", "48",
        "--batch_size", str(batch_size),
        "--alpha", f"{alpha}",
        "--learning_rate", str(lr),
        "--train_epochs", "100",
        "--patience", "10",
        "--itr", "1",
        "--pct_start", "0.2",
        "--seed", str(seed)
    ]

    # 执行命令并实时输出
    try:
        # 将stdout和stderr设为None，直接使用父进程的输出流
        result = subprocess.run(
            command,
            check=True,
            stdout=None,  # 实时输出到控制台
            stderr=None,  # 实时输出错误信息
            text=True
        )
        print(f"===== 参数组合执行成功: batch_size={batch_size}, learning_rate={lr}=====")
    except subprocess.CalledProcessError as e:
        print(
            f"===== 参数组合执行失败: batch_size={batch_size}, learning_rate={lr}, 返回码：{e.returncode} =====")