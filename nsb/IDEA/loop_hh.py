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
pred_len_list = args.lens
learning_rates_list = [0.006, 0.01, 5e-4]
comand_list = []
# 新增列表用于记录每次循环的信息
info_list = []
seed_list = [2020,2002,2003,2004,2005]
models_list = ["NSTS1", "NSTS2", "NSTS3"]
file_name = "all_IDEA"

for data in datalist:
    for model in models_list:
            for pred_len in pred_len_list:
                for lr in learning_rates_list:
                    for seed in seed_list:
                        seq_len = pred_len
                        label_len = 0
                        command = f"""     {d[data][str(pred_len)].replace("run.py", "run_nsts.py")}  --seq_len {seq_len}  --is_bn --label_len 0  --pred_len {pred_len} --gpu {args.device}    --batch_size {32}    --learning_rate {lr}     --train_epochs 40  --patience 5 --dropout 0.1 --seed {seed} """
                        comand_list.append(command)

i = 0
while i + args.size <= len(comand_list):
    new_comand = "".join(f"{comand_list[i + j]}  &  " for j in range(args.size)).rstrip().rstrip('&')
    os.system(new_comand)
    i = i + args.size

if i < len(comand_list):
    new_comand = "".join(f"{comand_list[i + j]}  &  " for j in range(len(comand_list) - i)).rstrip().rstrip('&')
    os.system(new_comand)
