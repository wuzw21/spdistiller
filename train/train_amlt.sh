
#export MODEL_PATH=NousResearch/Meta-Llama-3-8B
#export MODEL_NAME=Meta-Llama-3-8B
#export MODEL_PATH=NousResearch/Llama-2-13b-chat-hf
#export MODEL_NAME=Llama-2-13b-chat-hf

export DATA_PATH=$1
export SAVE_PATH=$2
export LOGGING_DIR=$3
export NUM_TRAIN_EPOCHS=$4
export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true

export MODEL_PATH=$5
export MODEL_NAME=$6

export PREDICT_DATASET_NAME=c4
export PREDICT_CKPT_HOME=${AMLT_DATA_DIR}/predictors/${MODEL_NAME}-${PREDICT_DATASET_NAME}
export ENABLE_PREDICTOR=1
export ENABLE_PREDICTOR_FINETUNE=0
export ENABLE_SPARSE_INFER=1
export PREDICTOR_DATA_DIR=${PREDICT_CKPT_HOME}
export ENABLE_TENSOR_SAVER=0

export NUM_NODES=$7
export NUM_GPUS=$8
export MAX_MEMORY=$9
#rm -rf /job/hostfile

# No ssh
#--hostfile=hostfile_remote --no_ssh --node_rank=0

# --clip BitDistiller/quantization/clip_cache/WizardCoder-7B/7b-int2-g128-twoclip.pt
# --evaluation_strategy "steps"
# --eval_steps 4
# --bits 4 --quant_type Q4_0 --q_group_size 64
deepspeed --num_nodes=${NUM_NODES} --num_gpus=${NUM_GPUS} \
    --hostfile=hostfile_local --no_ssh --node_rank=0 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} train.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --model_max_length 512 \
    --output_dir ${SAVE_PATH} \
    --logging_dir ${LOGGING_DIR} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --load_best_model_at_end False \
    --save_strategy "epoch" \
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
    --train_kd False \
    --kd_loss_type "cakld" \
    --max_train_samples 999999 \
    --max_memory ${MAX_MEMORY}
