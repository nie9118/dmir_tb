if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/TimeBridge" ]; then
    mkdir ./logs/LongForecasting/TimeBridge
fi

model_name=TimeBridge
seq_len=96
root=./data
gpu=0

alpha=0.35
data_name=ETTm1
for pred_len in 96 192 336 720
do
  HIP_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 7 \
    --ca_layers 0 \
    --pd_layers 1 \
    --ia_layers 3 \
    --des 'Exp' \
    --n_heads 4 \
    --d_model 64 \
    --d_ff 128 \
    --period 48 \
    --num_p 6 \
    --lradj 'TST' \
    --learning_rate 0.0002 \
    --train_epochs 100 \
    --pct_start 0.2 \
    --patience 15 \
    --batch_size 64 \
    --alpha $alpha \
    --itr 1
done

alpha=0.35
data_name=ETTm2
for pred_len in 96 192 336 720
do
  HIP_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/ETT-small/ \
    --data_path $data_name.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --pd_layers 1 \
    --enc_in 7 \
    --ca_layers 0 \
    --pd_layers 1 \
    --ia_layers 3 \
    --des 'Exp' \
    --n_heads 4 \
    --d_model 64  \
    --d_ff 128 \
    --lradj 'TST' \
    --period 48 \
    --train_epochs 100 \
    --learning_rate 0.0002 \
    --pct_start 0.2 \
    --patience 10 \
    --batch_size 64 \
    --alpha $alpha \
    --itr 1
done


alpha=0.1
data_name=weather
for pred_len in 96 192 336 720
do
  HIP_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/weather/ \
    --data_path weather.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 21 \
    --ca_layers 1 \
    --pd_layers 1 \
    --ia_layers 1 \
    --des 'Exp' \
    --period 48 \
    --num_p 12 \
    --d_model 128 \
    --d_ff 128 \
    --alpha $alpha \
    --itr 1
done


alpha=0.05
data_name=Solar
for pred_len in 96 192 336 720
do
  HIP_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 1 \
    --root_path $root/Solar/ \
    --data_path solar_AL.txt \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 137 \
    --ca_layers 1 \
    --pd_layers 1 \
    --ia_layers 1 \
    --des 'Exp' \
    --period 48 \
    --num_p 12 \
    --d_model 128 \
    --d_ff 128 \
    --alpha $alpha \
    --learning_rate 0.0005 \
    --train_epochs 100 \
    --patience 15 \
    --itr 1
done


alpha=0.2
GPU=0,1,2,3,4,5,6,7
data_name=electricity
for pred_len in 96 192 336 720
do
  MIOPEN_DISABLE_CACHE=1 \
  MIOPEN_SYSTEM_DB_PATH="" \
  HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  python -u run.py \
    --is_training 1 \
    --root_path $root/electricity/ \
    --data_path electricity.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 321 \
    --des 'Exp' \
    --n_heads 32 \
    --d_ff 512 \
    --d_model 512 \
    --ca_layers 2 \
    --pd_layers 1 \
    --ia_layers 1 \
    --attn_dropout 0.1 \
    --num_p 4 \
    --stable_len 4 \
    --alpha $alpha \
    --batch_size 16 \
    --devices 0,1,2,3,4,5,6,7 \
    --use_multi_gpu \
    --learning_rate 0.0005 \
    --itr 1
done

alpha=0.35
data_name=traffic
for pred_len in 336 720 192 96;
do
  MIOPEN_DISABLE_CACHE=1 \
  MIOPEN_SYSTEM_DB_PATH="" \
  HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  python -u run.py \
    --is_training 1 \
    --root_path $root/traffic/ \
    --data_path traffic.csv \
    --model_id $data_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --num_p 8 \
    --n_heads 64 \
    --stable_len 2 \
    --d_ff 512 \
    --d_model 512 \
    --ca_layers 3 \
    --pd_layers 1 \
    --ia_layers 1 \
    --batch_size 4 \
    --attn_dropout 0.15 \
    --patience 10 \
    --train_epochs 100 \
    --devices 0,1,2,3,4,5,6,7 \
    --use_multi_gpu \
    --alpha $alpha \
    --learning_rate 0.0005 \
    --itr 1
done
