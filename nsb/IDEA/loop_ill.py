import argparse
import os
from idea_config import d

parser = argparse.ArgumentParser()

parser.add_argument('-size', type=int, default=1)
parser.add_argument('-dataset', type=str, nargs='+')
parser.add_argument('-device', default=0, type=int)
parser.add_argument('-lens', default=0, type=int, nargs='+')

args = parser.parse_args()
datalist = args.dataset
pred_len_list = [96,192,336,720]
learning_rates_list = [0.00001,0.0001,0.0005,0.001]
batch_size_list = [16,32,64]
comand_list = []
# 新增列表用于记录每次循环的信息
info_list = []
seed_list = [2023]
file_name = "all_IDEA"
models_list = ["NSTS","NSTS1","NSTS2","NSTS3"]

for data in datalist:
    for model in models_list:
        for pred_len in pred_len_list:
            for lr in learning_rates_list:
                for seed in seed_list:
                    for batch_size in batch_size_list:
                        seq_len = 104
                        label_len = 0
                        command = f"""     {d[data][str(pred_len)].replace("run.py", "run_nsts.py")}  --model {model} --seq_len {seq_len}  --is_bn --label_len 0  --pred_len {pred_len} --gpu {args.device}    --batch_size {batch_size}    --learning_rate {lr}     --train_epochs 40  --patience 5 --seed {seed} """
                        comand_list.append(command)
                        # 记录当前循环的信息
                        info_list.append(f"当前模型：{model} 当前循环数据集：{data} 输入长度：{seq_len} 预测长度：{pred_len} learning_rate：{lr} seed:{seed}")

i = 0
while i + args.size <= len(comand_list):
    # 打印当前批次的第一个命令对应的信息
    print(info_list[i])
    new_comand = "".join(f"{comand_list[i + j]}  &  " for j in range(args.size)).rstrip().rstrip('&')
    os.system(new_comand)
    i = i + args.size

if i < len(comand_list):
    # 打印剩余命令的第一个信息
    print(info_list[i])
    new_comand = "".join(f"{comand_list[i + j]}  &  " for j in range(len(comand_list) - i)).rstrip().rstrip('&')
    os.system(new_comand)
