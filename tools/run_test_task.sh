#!/bin/bash

export MODEL_NAME=$1
# export MODEL_NAME=Phi-3.5-mini-instruct

MODEL=/data/wzw/models/${MODEL_NAME}

echo "MODEL path set to: $MODEL"

export ENABLE_PREDICTOR=1
export ENABLE_SPARSE_INFER=0
export ENABLE_TENSOR_SAVER=0

# unused parameters
export HF_HOME=${HOME}/Downloads/huggingface
export HF_DOWNLOAD_DATASET_HOME=${HF_HOME}/datasets
export HF_ENDPOINT=https://huggingface.co


export ATTN_SP=$2
export MLP_SP=$2
export W_P=0
export DO_CR=$3
SPARSE_STRATEGY=$4
export TEST_TASK=$5
export TEST_ALL=$6

if [ "$SPARSE_STRATEGY" = "Static" ]; then
    export THRESHOLD_PATH="../threshold/${MODEL_NAME}/sparse-${ATTN_SP}.json"
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
    --limit=5 \
    --num_shot=0 \
    --do_cr=${DO_CR} \
    --file_path=${THRESHOLD_PATH} \
    --sparse_strategy=${SPARSE_STRATEGY} \
    --test_all=${TEST_ALL}
cd ..
