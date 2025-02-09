#!/bin/bash

export HF_HOME=${HOME}/Downloads/huggingface
export MODEL_DIR=/data/wzw/models
export MODEL_NAME=Meta-Llama-3-8B
export MODEL=${MODEL_DIR}/${MODEL_NAME}
export TEST_TASK=mmlu
export CUDA_VISIBLE_DEVICES=0
export ENABLE_PREDICTOR=1
export ENABLE_SPARSE_INFER=1
export ENABLE_TENSOR_SAVER=0
export PREDICTOR_DATA_HOME=${HF_HOME}/predictor-data
export PREDICTOR_DATA_DIR=${PREDICTOR_DATA_HOME}/${MODEL_NAME}-c4-sparse
export PREDICT_CKPT_HOME=/data/fuchengjia/Projects/llm-wanda/checkpoints/weight-predictors

export LOCAL_RANK=-1

echo "Model: ${MODEL_NAME}"

echo "CUDA device: ${CUDA_VISIBLE_DEVICES}"

cd train

#--limit=500
python test_chat.py 