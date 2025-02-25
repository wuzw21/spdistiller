#!/bin/bash

export MODEL_DIR=/data/wzw/models
export MODEL_NAME=Meta-Llama-3-8B
export MODEL=${MODEL_DIR}/${MODEL_NAME}
export TEST_TASK=mmlu
export CUDA_VISIBLE_DEVICES=0

export ENABLE_PREDICTOR=1
export ENABLE_SPARSE_INFER=1
export ENABLE_TENSOR_SAVER=0

export ATTN_SP=0.7
export MLP_SP=0.7
export W_P=0.0
export DO_CR=1


export LOCAL_RANK=-1

echo "Model: ${MODEL_NAME}"

echo "CUDA device: ${CUDA_VISIBLE_DEVICES}"

cd train

#--limit=500
python test_chat.py 