#!/bin/bash

export MODEL_DIR=/data/wzw/models/Mixtral
export MODEL_NAME=Mixtral-8x7B-v0.1
export MODEL=${MODEL_DIR}/${MODEL_NAME}
export TEST_TASK=wiki
export CUDA_VISIBLE_DEVICES=0

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