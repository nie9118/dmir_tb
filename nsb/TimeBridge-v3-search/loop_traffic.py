import subprocess
import os
from itertools import product

# 设置环境变量（指定GPU）
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# 配置基础参数
model_name = "TimeBridge"
data_name = "traffic"
root='./data' # 数据集根路径
data_path = 'traffic' # 可选[ETT-small，electricity，exchange_rate，illness，traffic，weather]
seq_len=720
pred_len=720
alpha=0.35

enc_in=862

# 定义要搜索的参数网格
batch_sizes = [16]
learning_rates = [0.0002]
ca_layers = [3,4,5,6]  # 长期
pd_layers = [1]
ia_layers = [0,1,2,3]  # 短期

# 生成所有参数组合
param_combinations = product(batch_sizes, learning_rates, ca_layers, pd_layers, ia_layers)

# 遍历每个参数组合并执行命令
for batch_size, lr, ca_layers, pd_layers, ia_layers in param_combinations:
    print(f"\n===== 开始执行参数组合: batch_size={batch_size}, learning_rate={lr}=====")

    # 构建命令列表
    command = [
        "python", "run.py",
        "--is_training", "1",
        "--root_path",f"{root}/{data_path}/",
        "--data_path",f"{data_name}.csv",
        "--model_id",f"{data_name}'_'{seq_len}'_'{pred_len}",
        "--model",f"{model_name}",
        "--data",f"custom",
        "--features","M",
        "--seq_len",f"{seq_len}",
        "--label_len","48",
        "--pred_len",f"{pred_len}",
        "--enc_in",f"{enc_in}",
        "--des","Exp",
        "--num_p","8",
        "--n_heads","64",
        "--stable_len","2",
        "--d_ff","512",
        "--d_model","512",
        "--ca_layers",str(ca_layers),
        "--pd_layers",str(pd_layers),
        "--ia_layers",str(ia_layers),
        "--batch_size",str(batch_size),
        "--attn_dropout","0.15",
        "--devices","0,1",
        "--use_multi_gpu",
        "--alpha",f"{alpha}",
        "--patience","5",
        "--learning_rate",str(lr),
        "--train_epochs","100",
        "--itr","1"
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