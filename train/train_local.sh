# !/bin/bash
export DATA_PATH=$1
export SAVE_PATH=$2
export LOGGING_DIR=$3

export WANDB_DISABLED=true

# No ssh
#--hostfile=hostfile_remote --no_ssh --node_rank=0

# --clip BitDistiller/quantization/clip_cache/WizardCoder-7B/7b-int2-g128-twoclip.pt
# --evaluation_strategy "steps"
# --eval_steps 4
# --bits 4 --quant_type Q4_0 --q_group_size 64
python train_single.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --threshold_path ${THRESHOLD_PATH} \
    --model_max_length 512 \
    --output_dir ${SAVE_PATH} \
    --logging_dir ${LOGGING_DIR} \
    --num_train_epochs 5 \
    --seed 42 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --load_best_model_at_end False \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --weight_decay 0. \
    --logging_steps 1 \
    --report_to "none" \
    --bits 4 \
    --quant_type Q4_0 \
    --q_group_size 64 \
    --train_kd False \
    --kd_loss_type "reverse" \
    --evaluation_strategy "steps" \
    --eval_steps  100 \
    --use_lora True