if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/TimeBridge_v3_test" ]; then
    mkdir ./logs/LongForecasting/TimeBridge_v3_test
fi

model_name=TimeBridge
seq_len=720
GPU=1
root=./dataset
pred_len=720

alpha=0.35
data_name=ETTh1
for seed in $(seq 1 100)
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u run.py \
    --seed $seed \
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
    --ca_layers 1 \
    --pd_layers 1 \
    --ia_layers 3 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 16 \
    --alpha $alpha \
    --learning_rate 0.00013306254818603393 \
    --train_epochs 100 \
    --patience 10 \
    --n_heads 64 \
    --itr 1 | tee logs/LongForecasting/TimeBridge_v3_test/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

#alpha=0.35
#data_name=ETTh1
#for pred_len in 48 96 144 192
#do
#  seq_len=$((2 * pred_len))
#  CUDA_VISIBLE_DEVICES=$GPU \
#  python -u tune.py \
#    --is_training 1 \
#    --root_path $root/ETT-small/ \
#    --data_path $data_name.csv \
#    --model_id $data_name'_'$seq_len'_'$pred_len \
#    --model $model_name \
#    --data $data_name \
#    --features M \
#    --seq_len $seq_len \
#    --label_len 48 \
#    --pred_len $pred_len \
#    --enc_in 7 \
#    --ca_layers 0 \
#    --pd_layers 1 \
#    --ia_layers 3 \
#    --des 'Exp' \
#    --d_model 128 \
#    --d_ff 128 \
#    --batch_size 64 \
#    --alpha $alpha \
#    --learning_rate 0.0002 \
#    --train_epochs 100 \
#    --patience 10 \
#    --itr 1 | tee logs/LongForecasting/TimeBridge_v3_test/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
#    done

#alpha=0.35
#data_name=ETTh2
#for pred_len in 336 720
#do
#  CUDA_VISIBLE_DEVICES=$GPU \
#  python -u tune.py \
#    --is_training 1 \
#    --root_path $root/ETT-small/ \
#    --data_path $data_name.csv \
#    --model_id $data_name'_'$seq_len'_'$pred_len \
#    --model $model_name \
#    --data $data_name \
#    --features M \
#    --seq_len $seq_len \
#    --label_len 48 \
#    --pred_len $pred_len \
#    --enc_in 7 \
#    --period 48 \
#    --ca_layers 0 \
#    --pd_layers 1 \
#    --ia_layers 3 \
#    --ca_layers 0 \
#    --des 'Exp' \
#    --n_heads 4 \
#    --period 48 \
#    --d_model 128 \
#    --d_ff 128 \
#    --train_epochs 100 \
#    --learning_rate 0.0001 \
#    --patience 15 \
#    --alpha $alpha \
#    --batch_size 16 \
#    --itr 1 | tee logs/LongForecasting/TimeBridge_v3_test/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
#done

#alpha=0.35
#data_name=ETTh2
#for pred_len in 96 192 336 720
#do
#  seq_len=$((2 * pred_len))
#  CUDA_VISIBLE_DEVICES=$GPU \
#  python -u tune.py \
#    --is_training 1 \
#    --root_path $root/ETT-small/ \
#    --data_path $data_name.csv \
#    --model_id $data_name'_'$seq_len'_'$pred_len \
#    --model $model_name \
#    --data $data_name \
#    --features M \
#    --seq_len $seq_len \
#    --label_len 48 \
#    --pred_len $pred_len \
#    --enc_in 7 \
#    --period 48 \
#    --ca_layers 0 \
#    --pd_layers 1 \
#    --ia_layers 3 \
#    --ca_layers 0 \
#    --des 'Exp' \
#    --n_heads 4 \
#    --period 48 \
#    --d_model 128 \
#    --d_ff 128 \
#    --train_epochs 100 \
#    --learning_rate 0.0001 \
#    --patience 15 \
#    --alpha $alpha \
#    --batch_size 16 \
#    --itr 1 | tee logs/LongForecasting/TimeBridge_v3_test/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
#done