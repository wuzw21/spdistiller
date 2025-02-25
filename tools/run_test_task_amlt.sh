#!/bin/bash

export MODEL_NAME=$1
# export MODEL_NAME=Phi-3.5-mini-instruct
export TEST_TASK=mmlu


# 检查 $4 是否有值
if [ -n "$4" ]; then
    # 如果 $4 有值，则使用 $4
    export MODEL=$4
else
    # 如果 $4 为空，则按照原逻辑处理
    if [ -n "$AMLT_MAP_INPUT_DIR" ]; then
        # 如果 AMLT_MAP_INPUT_DIR 存在，则使用它
        export MODEL=${AMLT_MAP_INPUT_DIR}/ckpts/${MODEL_NAME}/int4-g64
    else
        # 如果 AMLT_MAP_INPUT_DIR 不存在，则使用 AMLT_DATA_DIR
        export MODEL=${AMLT_DATA_DIR}/models/${MODEL_NAME}
    fi
fi

echo "MODEL path set to: $MODEL"
# 
# export MODEL=/data/wzw/models/Llama-3.1-8B-Instruct
export ENABLE_PREDICTOR=1
export ENABLE_SPARSE_INFER=1
export ENABLE_TENSOR_SAVER=0

# unused parameters
export HF_HOME=${HOME}/Downloads/huggingface
export HF_DOWNLOAD_DATASET_HOME=${HF_HOME}/datasets
export HF_ENDPOINT=https://huggingface.co


export ATTN_SP=$2
export MLP_SP=$2
export W_P=0
export DO_CR=$3
SPARSE_STRATEGY="Static"
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
