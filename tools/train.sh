#!/bin/bash

cd train

DATA_PATH=${DATA_DIR}/datasets/${MODEL_NAME}/${DATASET} 
SAVE_PATH=${OUTPUT_DIR}/ckpts/${MODEL_NAME}
LOGGING_DIR=${OUTPUT_DIR}/logs/${MODEL_NAME}
echo "SAVE_PATH" $SAVE_PATH
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
deepspeed --no_ssh --node_rank=0  \
    --num_nodes=1 --num_gpus=4 \
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
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing False \
    --load_best_model_at_end False \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --logging_steps 1 \
    --learning_rate 2e-4 \
    --report_to "none" \
    --quant ${QUANT} \
    --bits 4 \
    --quant_type Q4_0 \
    --q_group_size 32 \
    --train_kd False \
    --kd_loss_type "forward" \
    --max_train_samples 999999 \
    --evaluation_strategy "epoch" \
    --use_lora ${USE_LORA} \
    --load_checkpoint False

cd ..
 