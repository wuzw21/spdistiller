#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.3


export MODEL_DIR=/data/wzw/models
export MODEL_NAME=$1
export MODEL=${MODEL_DIR}/${MODEL_NAME}

export TEST_TASK=wiki

export ENABLE_PREDICTOR=1
export ENABLE_SPARSE_INFER=1
export ENABLE_TENSOR_SAVER=0

export HF_HOME=${HOME}/Downloads/huggingface
export HF_DOWNLOAD_DATASET_HOME=${HF_HOME}/datasets
export HF_ENDPOINT=https://huggingface.co


export ATTN_SP=$2
export MLP_SP=$2
export W_P=0
export DO_CR=$3
SPARSE_STRATEGY='Static'

if [ "$SPARSE_STRATEGY" = "Static" ]; then
    export THRESHOLD_PATH="../threshold/${MODEL_NAME}/${MODEL_NAME}-${ATTN_SP}.txt"
else
    export THRESHOLD_PATH="zwwz"
fi


echo "Model: ${MODEL_NAME}"

echo "CUDA device: ${CUDA_VISIBLE_DEVICES}"

echo "Model: ${MODEL}"

cd train

#--limit=500
python test_task.py \
    --model=${MODEL} \
    --seed=42 \
    --task=${TEST_TASK} \
    --sparse=${ATTN_SP} \
    --limit=100 \
    --num_shot=0 \
    --do_cr=${DO_CR} \
    --file_path=${THRESHOLD_PATH} \
    --sparse_strategy=${SPARSE_STRATEGY}
cd ..
