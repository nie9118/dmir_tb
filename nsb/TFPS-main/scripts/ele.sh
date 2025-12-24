if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=PatchTST_MoE_cluster

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom

random_seed=2021
for seq_len in 720
do
for pred_len in 96 192 336 720
do
for random_seed in 2021
do
for learning_rate in 0.001 0.0001 0.0005 0.001 0.005 0.01 0.05
do
for T_num_expert in 1 2 4 8
do
for T_top_k in 1 2 4 8
do
for F_num_expert in 1 2 4 8
do
for F_top_k in 1 2 4 8
do
    # AMD GPU环境变量配置
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export MIOPEN_DISABLE_CACHE=1
    export MIOPEN_SYSTEM_DB_PATH=''

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --e_layers 3 \
      --n_heads 32 \
      --d_model 512 \
      --d_ff 512 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --T_num_expert $T_num_expert\
      --T_top_k $T_top_k\
      --F_num_expert $F_num_expert\
      --F_top_k $F_top_k\
      --beta 0.1 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'TST'\
      --use_multi_gpu \
      --devices 0,1,2,3,4,5,6,7 \
      --pct_start 0.4\
      --itr 1 --batch_size 128 --learning_rate $learning_rate
done
done
done
done
done
done
done
done