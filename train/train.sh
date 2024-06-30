
export HF_HOME=${HOME}/Downloads/Huggingface
export MODEL_PATH=${HF_HOME}/Models/Llama-2-7b-chat-hf
export MODEL_NAME=Llama-2-7b-chat-hf
export SAVE_PATH=$2
export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true

export PREDICT_DATASET_NAME=c4
export PREDICT_CKPT_HOME=${HOME}/Projects/llm-wanda/checkpoints/predictors/${MODEL_NAME}-${PREDICT_DATASET_NAME}
export ENABLE_PREDICTOR=1
export ENABLE_PREDICTOR_FINETUNE=0

# --clip BitDistiller/quantization/clip_cache/WizardCoder-7B/7b-int2-g128-twoclip.pt
deepspeed --num_gpus=1 train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $1 \
    --model_max_length 1024 \
    --output_dir $SAVE_PATH \
    --logging_dir $3 \
    --num_train_epochs $4 \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 4 \
    --load_best_model_at_end False \
    --save_strategy "steps" \
    --save_steps 4 \
    --save_total_limit 1 \
    --learning_rate 8e-6 \
    --lr_scheduler_type "constant" \
    --weight_decay 0. \
    --logging_steps 1 \
    --report_to "none" \
    --deepspeed config/zero.json \
    --bits 4 \
    --quant_type Q4_0 \
    --q_group_size 64 \
    --train_kd True \
    --kd_loss_type "cakld" \
    --max_train_samples 999999
