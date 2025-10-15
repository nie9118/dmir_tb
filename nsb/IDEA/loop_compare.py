import argparse
import os
from compare_config import d

parser = argparse.ArgumentParser()

parser.add_argument('-models', type=str, nargs='+', required=True)
parser.add_argument('-size', type=int, default=1)
parser.add_argument('-dataset', type=str, nargs='+')
parser.add_argument('-device', default=0, type=int)
# parser.add_argument('-lens', default=0, type=int, nargs='+')

args = parser.parse_args()
datalist = args.dataset
comand_list = []
type_list = ['type1']
seed_list = [2024]
# seed_list = [2022]
pred_len_list = [48, 96, 144, 192]
file_name = "IDEA"
# file_name = "time"

for data in datalist:
    for pred_len in pred_len_list:
        for type in type_list:
            for seed in seed_list:
                for model in args.models:
                    seq_len = 96
                    if model == "MICN":
                        label_len = pred_len * 2
                    elif model == 'WITRAN' or model == 'FAN' or model == 'Foil':
                        label_len = 0
                    else:
                        label_len = int(pred_len * 3 // 2)
                    command = f'python ./run_nsts.py   --gpu {args.device} --checkpoints ./{file_name}_result   --batch_size {32}      --model {model}   --lradj {type}  {d[data][model]} ' \
                              f' --seq_len {seq_len} --label_len {label_len}  --pred_len {pred_len}  --train_epochs 15  --patience 5 --seed {seed} '
                    comand_list.append(command)

i = 0
while i + args.size <= len(comand_list):
    new_comand = "".join(f"{comand_list[i + j]}  &  " for j in range(args.size)).rstrip().rstrip('&')
    os.system(new_comand)
    i = i + args.size

if i < len(comand_list):
    new_comand = "".join(f"{comand_list[i + j]}  &  " for j in range(len(comand_list) - i)).rstrip().rstrip('&')
    os.system(new_comand)
