
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=${CUDA_HOME}/targets/x86_64-linux/lib/stubs:${PATH}
#export LD_LIBRARY_PATH=${HOME}/Software/miniconda3/envs/py310/lib/python3.10/site-packages/nvidia/curand/lib:${LD_LIBRARY_PATH}
export MAX_JOBS=16

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
export PREDICT_CKPT_HOME=${HF_HOME}/predictor-data/${MODEL_NAME}-${PREDICT_DATASET_NAME}
export ENABLE_PREDICTOR=1
export ENABLE_PREDICTOR_FINETUNE=0
export ENABLE_SPARSE_INFER=1
export PREDICTOR_DATA_DIR=${PREDICT_CKPT_HOME}
export ENABLE_TENSOR_SAVER=0

export NUM_GPUS=$7

# --clip BitDistiller/quantization/clip_cache/WizardCoder-7B/7b-int2-g128-twoclip.pt
# --evaluation_strategy "steps"
# --eval_steps 4
deepspeed --hostfile=hostfile --no_ssh --node_rank=0 \
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
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --load_best_model_at_end False \
    --save_strategy "steps" \
    --save_steps 100 \
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
    --max_train_samples 999999
