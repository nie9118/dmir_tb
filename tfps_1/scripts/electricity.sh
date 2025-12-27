if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/electricity" ]; then
    mkdir ./logs/LongForecasting/electricity
fi

# Electricity dataset uses the generic custom CSV loader.
# Assumptions (adjust to your actual file):
# - CSV has a 'date' column
# - all other columns are variables/features
# - multivariate forecasting: features=M

model_name=PatchTST_MoE_cluster

GPU=0,1,2,3
root_path_name=../dataset/electricity/
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom

export CUDA_VISIBLE_DEVICES=$GPU

# Keep the same fast config style as solar.sh
BATCH_SIZE=2
D_MODEL=32
N_HEADS=4
E_LAYERS=3
D_FF=128
PATCH_LEN=16
STRIDE=8
T_NUM_EXPERT=1
T_TOP_K=2
F_NUM_EXPERT=1
F_TOP_K=2
LR=0.0003
TRAIN_EPOCHS=30
DROPOUT=0.2
FC_DROPOUT=0.2
PATIENCE=10
LRADJ='TST'

# IMPORTANT: set ENC_IN/C_OUT to the number of variables in your CSV (excluding 'date').
# The public Electricity dataset is commonly 321 variables.
# If yours differs, change ENC_IN/C_OUT.
ENC_IN=321
C_OUT=321
FREQ=h

# PatchTST-style setting
LABEL_LEN=48

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
      --devices 0,1,2,3 \
      --use_multi_gpu \
      --itr 1 --batch_size ${BATCH_SIZE} --learning_rate ${learning_rate}
done
done
done
done
done
done
done
done

echo "All experiments submitted"
