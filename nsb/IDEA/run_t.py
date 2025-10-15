import argparse
import os
from new_config import c

parser = argparse.ArgumentParser()

parser.add_argument('-size', type=int, default=1)
parser.add_argument('-dataset', type=str, nargs='+')
parser.add_argument('-device', default=0, type=int)

args = parser.parse_args()
datalist = args.dataset
pred_len_list = [96,192,336,720]
comand_list = []
# 新增列表用于记录每次循环的信息
info_list = []
seed_list = [2023]
file_name = "all_IDEA"
models_list = ["NSTS2","NSTS3"]

for data in datalist:
    for model in models_list:
        for pred_len in pred_len_list:
             for seed in seed_list:
                    seq_len = 96
                    label_len = 0
                    command = f"""     {c[data][str(pred_len)].replace("run.py", "tune.py")}  --model {model} --seq_len {seq_len}  --is_bn --label_len 0  --pred_len {pred_len} --gpu {args.device}   --train_epochs 40  --patience 5 --seed {seed} """
                    comand_list.append(command)
                    # 记录当前循环的信息
                    info_list.append(f"当前模型：{model} 当前循环数据集：{data} 输入长度：{seq_len} 预测长度：{pred_len}  seed:{seed}")

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
