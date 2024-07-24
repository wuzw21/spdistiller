
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=${CUDA_HOME}/targets/x86_64-linux/lib/stubs:${PATH}
#export LD_LIBRARY_PATH=${HOME}/Software/miniconda3/envs/py310/lib/python3.10/site-packages/nvidia/curand/lib:${LD_LIBRARY_PATH}
export MAX_JOBS=16

export HF_HOME=/data/fuchengjia/Downloads/huggingface
export MODEL_PATH=${HF_HOME}/models/Meta-Llama-3-8B
export MODEL_NAME=Meta-Llama-3-8B

export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true

export PREDICT_DATASET_NAME=c4
export PREDICT_CKPT_HOME=${HF_HOME}/predictor-data/${MODEL_NAME}-${PREDICT_DATASET_NAME}
export ENABLE_PREDICTOR=0
export ENABLE_PREDICTOR_FINETUNE=0

cd train

python test_infer.py \
    --model_name_or_path ${MODEL_PATH} \
    --model_max_length 512 \
    --quant_type Q4_0 \
    --q_group_size 64 \
    --data_path ../data/datasets/${MODEL_NAME}/c4_T0.7_N1024_S42_4096.json \
    --per_device_train_batch_size 4 \
    --output_dir /tmp/bitdistiller
