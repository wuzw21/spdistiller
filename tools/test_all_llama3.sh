#!/bin/bash

# TODO : fix all Models
MODELS=(
    # "/mnt/default/projects/wzw-gcrbitdistillerq4/amlt-results/7260183513.51709-a9f15020-08d9-4430-8043-efd351b3a4c6/ckpts/Meta-Llama-3-8B/int4-g64"
    "/mnt/default/projects/wzw-gcrbitdistillerq4/amlt-results/7260792207.26018-730d80a0-26bb-4fa4-9aa2-2874c8933463/ckpts/Meta-Llama-3-8B/int4-g64"
)
SPARSES=(
    # 0.8
    0.7
)

export MODEL_NAME="Meta-Llama-3-8B"
export TEST_TASK="mmlu"

export ENABLE_PREDICTOR=1
export ENABLE_SPARSE_INFER=1
export ENABLE_TENSOR_SAVER=0

export HF_HOME="${HOME}/Downloads/huggingface"
export HF_DOWNLOAD_DATASET_HOME="${HF_HOME}/datasets"
export HF_ENDPOINT="https://huggingface.co"

export SPARSE_STRATEGY='Static'

# 使用 bash 数组索引迭代
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    SPARSE="${SPARSES[$i]}"
    DO_CR=0
    bash tools/run_test_task_amlt.sh "${MODEL_NAME}" "${SPARSE}" "${DO_CR}" "${SPARSE_STRATEGY}" "${MODEL}"
done