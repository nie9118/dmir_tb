if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/TimeBridge_v3" ]; then
    mkdir ./logs/LongForecasting/TimeBridge_v3
fi
model_name=TimeBridge
GPU=1
root=./dataset
seq_len=720

alpha=0.1
data_name=weather
for pred_len in 96 192 336 720
do
  CUDA_VISIBLE_DEVICES=$GPU \
  python -u tune.py \
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
    --itr 1 | tee logs/LongForecasting/TimeBridge_v3/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
done

#alpha=0.1
#data_name=weather
#for pred_len in 48 96 144 192
#do
#  seq_len=$((2 * pred_len))
#  CUDA_VISIBLE_DEVICES=$GPU \
#  python -u tune.py \
#    --is_training 1 \
#    --root_path $root/weather/ \
#    --data_path weather.csv \
#    --model_id $data_name'_'$seq_len'_'$pred_len \
#    --model $model_name \
#    --data custom \
#    --features M \
#    --seq_len $seq_len \
#    --label_len 48 \
#    --pred_len $pred_len \
#    --enc_in 21 \
#    --ca_layers 1 \
#    --pd_layers 1 \
#    --ia_layers 1 \
#    --des 'Exp' \
#    --period 48 \
#    --num_p 12 \
#    --d_model 128 \
#    --d_ff 128 \
#    --alpha $alpha \
#    --itr 1 | tee logs/LongForecasting/TimeBridge3/$data_name'_'$alpha'_'$model_name'_'$pred_len.logs
#done
