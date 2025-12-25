if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/weather" ]; then
    mkdir ./logs/LongForecasting/weather
fi

# Weather dataset typically uses the generic custom CSV loader.
# Assumptions (adjust if your file differs):
# - CSV has a 'date' column
# - all other columns are features
# - multivariate forecasting: features=M

model_name=PatchTST_MoE_cluster

GPU=0,1,2,3,4,5,6,7
root_path_name=../dataset/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

export MIOPEN_DISABLE_CACHE=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export HIP_VISIBLE_DEVICES=$GPU

# Keep the same fast config style as solar.sh
# Recommended config for Weather (21 vars, 10-min frequency):
BATCH_SIZE=32
D_MODEL=64
N_HEADS=8
E_LAYERS=3
D_FF=256
PATCH_LEN=16
STRIDE=8
T_NUM_EXPERT=8
T_TOP_K=2
F_NUM_EXPERT=8
F_TOP_K=2
LR=0.0003
TRAIN_EPOCHS=30
DROPOUT=0.2
FC_DROPOUT=0.2
PATIENCE=10
LRADJ='TST'
LABEL_LEN=48

# IMPORTANT: set ENC_IN/C_OUT to the number of variables in your CSV (excluding 'date').
# Weather dataset is commonly 21 variables. If yours differs, change ENC_IN/C_OUT.
ENC_IN=21
C_OUT=21
# Your weather.csv is 10-minute data; in pandas diff it's 0 days 00:10:00.
# Use a matching freq string so time feature encoding is sensible.
FREQ=10min

for seq_len in 720
do
for pred_len in 96 192 336 720
do
for random_seed in 2023
do
for learning_rate in ${LR}
do
for T_num_expert in ${T_NUM_EXPERT}
do
for T_top_k in ${T_TOP_K}
do
for F_num_expert in ${F_NUM_EXPERT}
do
for F_top_k in ${F_TOP_K}
do
    MIOPEN_DISABLE_CACHE=1 \
    MIOPEN_SYSTEM_DB_PATH="" \
    HIP_VISIBLE_DEVICES="$GPU" \
    python -u ../run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --target 0 \
      --freq ${FREQ} \
      --seq_len $seq_len \
      --label_len ${LABEL_LEN} \
      --pred_len $pred_len \
      --enc_in ${ENC_IN} \
      --c_out ${C_OUT} \
      --e_layers ${E_LAYERS} \
      --n_heads ${N_HEADS} \
      --d_model ${D_MODEL} \
      --d_ff ${D_FF} \
      --dropout ${DROPOUT} \
      --fc_dropout ${FC_DROPOUT} \
      --head_dropout 0 \
      --patch_len ${PATCH_LEN} \
      --stride ${STRIDE} \
      --T_num_expert $T_num_expert \
      --T_top_k $T_top_k \
      --F_num_expert $F_num_expert \
      --F_top_k $F_top_k \
      --beta 0.01 \
      --des 'Exp' \
      --train_epochs ${TRAIN_EPOCHS} \
      --patience ${PATIENCE} \
      --lradj ${LRADJ} \
      --devices 0,1,2 \
      --use_gpu True \
      --gpu 1 \
      --itr 1 --batch_size ${BATCH_SIZE} --learning_rate ${learning_rate}  | tee logs/LongForecasting/weather/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${T_num_expert}_${T_top_k}_${F_num_expert}_${F_top_k}_${learning_rate}_0.1.log

done
done
done
done
done
done
done
done

echo "All experiments submitted"
