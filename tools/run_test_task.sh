#!/bin/bash

#export HF_HOME=${HOME}/Downloads/huggingface
export HF_HOME=/home/donglinbai/Projects/wzw

export MODEL_DIR=${HF_HOME}/models
#export MODEL_DIR=${AMLT_DATA_DIR}/models
export MODEL_NAME=$1

export MODEL=${MODEL_DIR}/${MODEL_NAME}

export ENABLE_PREDICTOR=1
export ENABLE_SPARSE_INFER=1
export ENABLE_TENSOR_SAVER=0
export PREDICTOR_DATA_HOME=${HF_HOME}/predictor-data
export PREDICTOR_DATA_DIR=${PREDICTOR_DATA_HOME}/${MODEL_NAME}-c4-sparse
export PREDICT_CKPT_HOME=/data/fuchengjia/Projects/llm-wanda/checkpoints/weight-predictors

export LOCAL_RANK=-1

export PROSPARSE_PREDICTOR=0
export PROSPARSE_PREDICTOR_DIR=${HF_HOME}/models/${MODEL_NAME}-predictor

export HF_DOWNLOAD_DATASET_HOME=${HF_HOME}/datasets
export HF_ENDPOINT=https://huggingface.co

echo "Model: ${MODEL_NAME}"

echo "CUDA device: ${CUDA_VISIBLE_DEVICES}"

cd test

#--limit=500
python test_task.py \
    --model=${MODEL} \
    --seed=42 \
    --task=$2

cd ..
