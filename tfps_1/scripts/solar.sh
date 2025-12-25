if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/solar" ]; then
    mkdir ./logs/LongForecasting/solar
fi


# seq_len=96
model_name=PatchTST_MoE_cluster

GPU=0,1,2,3
root_path_name=../dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=solar
data_name=solar
export CUDA_VISIBLE_DEVICES=$GPU
# random_seed=2023

# Optimized training config for Solar + PatchTST_MoE_cluster (Faster!)
BATCH_SIZE=96          # 增加batch size (24→32) 提高GPU利用率
D_MODEL=8              # 大幅减小d_model (16→8) 减少参数量
N_HEADS=4              # 减少注意力头 (8→4) 
E_LAYERS=2             # 减少层数 (3→2) 加速训练
D_FF=32                # 减小FFN维度 (64→32)
PATCH_LEN=16
STRIDE=8
T_NUM_EXPERT=8         # 减少专家数 (16→8) 关键优化!
T_TOP_K=1
F_NUM_EXPERT=8         # 减少专家数 (16→8) 关键优化!
F_TOP_K=1
LR=0.0005              # 增大学习率 (0.00001→0.0001) 加速收敛
TRAIN_EPOCHS=100        # 减少训练轮数 (100→50)
DROPOUT=0.1
FC_DROPOUT=0.1

for seq_len in 720
do
for pred_len in 720
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
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 137 \
      --c_out 137 \
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
      --devices 0,1,2,3 \
      --use_multi_gpu \
      --itr 1 --batch_size ${BATCH_SIZE} --learning_rate ${learning_rate}  | tee logs/LongForecasting/solar/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${T_num_expert}_${T_top_k}_${F_num_expert}_${F_top_k}_${learning_rate}_0.1.log

done
done
done
done
done
done
done
done

echo "All experiments submitted"