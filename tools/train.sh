#!/bin/bash

cd train

DATA_PATH=${DATA_DIR}/datasets/${MODEL_NAME}/${DATASET} 
SAVE_PATH=${OUTPUT_DIR}/ckpts/${MODEL_NAME}
LOGGING_DIR=${OUTPUT_DIR}/logs/${MODEL_NAME}

export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true

# No ssh
#--hostfile=hostfile_remote --no_ssh --node_rank=0

# --clip BitDistiller/quantization/clip_cache/WizardCoder-7B/7b-int2-g128-twoclip.pt
# --evaluation_strategy "steps"
# --eval_steps 4
# --bits 4 --quant_type Q4_0 --q_group_size 64
echo $DATA_PATH
deepspeed --no_ssh --node_rank=0 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} train.py \
    --deepspeed config/zero.json \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --threshold_path ${THRESHOLD_PATH} \
    --model_max_length 512 \
    --output_dir ${SAVE_PATH} \
    --logging_dir ${LOGGING_DIR} \
    --num_train_epochs 5 \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing False \
    --load_best_model_at_end False \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --weight_decay 0. \
    --logging_steps 1 \
    --learning_rate 5e-5 \
    --report_to "none" \
    --bits 4 \
    --quant_type Q4_0 \
    --q_group_size 64 \
    --train_kd True \
    --kd_loss_type "forward" \
    --max_train_samples 999999 \
    --evaluation_strategy "epoch" \
    --use_lora False

cd ..
