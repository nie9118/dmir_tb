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
comand_list = []

seed_list = [2023]
pred_len_list = args.lens
file_name = "loop_cs"
drop_out_list = [0, 0.2, 0.5]
hidden_dim_list = [128, 256, 384, 512, 640, 768, 1024]
hidden_layers_list = [1, 2, 3]
activation_list = ["ide", "relu"]
for data in datalist:

    for pred_len in pred_len_list:
        for seed in seed_list:
            for hidden_layers in hidden_layers_list:

                for hidden_dim in hidden_dim_list:
                    for dropout in drop_out_list:
                        for activation in activation_list:
                            seq_len = pred_len * 3
                            label_len = 0
                            command = f"""     {d[data][str(pred_len)].replace("run.py", "run_nsts.py")}    --checkpoints ./{file_name}   --gpu {args.device}    --batch_size {32}      --train_epochs 15  --patience 5 --seed {seed}  --hidden_dim {hidden_dim} --hidden_layers  {hidden_layers}  --dropout {dropout}  --activation {activation} --No_prior """
                            comand_list.append(command)
                            if hidden_layers == 1:
                                break
                        if hidden_layers == 1:
                            break
                    if hidden_layers == 1:
                        break

i = 0
while i + args.size <= len(comand_list):
    new_comand = "".join(f"{comand_list[i + j]}  &  " for j in range(args.size)).rstrip().rstrip('&')
    os.system(new_comand)
    i = i + args.size

if i < len(comand_list):
    new_comand = "".join(f"{comand_list[i + j]}  &  " for j in range(len(comand_list) - i)).rstrip().rstrip('&')
    os.system(new_comand)
