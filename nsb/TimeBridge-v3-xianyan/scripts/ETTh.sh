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
GPU=0
root=./data

alpha=0.322251067
data_name=ETTh2
for pred_len in 96
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --is_training 0 \
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
    --ca_layers 1 \
    --pd_layers 1 \
    --ia_layers 2 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 32 \
    --alpha $alpha \
    --learning_rate 0.000341653 \
    --train_epochs 100 \
    --patience 10 \
    --n_head 16 \
    --period 48 \
    --patience 15 \
    --itr 1
done