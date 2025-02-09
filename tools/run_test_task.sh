#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.3
export HF_HOME=${HOME}/Downloads/huggingface
export MODEL_DIR=/data/wzw/models
export FILE_PATH=../threshold/llama-3-0.7.txt
export MODEL_NAME=Meta-Llama-3-8B
export MODEL=/home/donglinbai/Projects/wzw/BitDistiller-Q4_0/data/ckpts/Meta-Llama-3-8B/int4-g64/checkpoint-1818
export TEST_TASK=wiki
export ENABLE_PREDICTOR=1
export ENABLE_SPARSE_INFER=1
export ENABLE_TENSOR_SAVER=0
export PREDICTOR_DATA_HOME=${HF_HOME}/predictor-data
export PREDICTOR_DATA_DIR=${PREDICTOR_DATA_HOME}/${MODEL_NAME}-c4-sparse
export PREDICT_CKPT_HOME=/data/fuchengjia/Projects/llm-wanda/checkpoints/weight-predictors
export TOKENIZERS_PARALLELISM=false
export LOCAL_RANK=-1

export PROSPARSE_PREDICTOR=0
export PROSPARSE_PREDICTOR_DIR=${HF_HOME}/models/${MODEL_NAME}-predictor

export HF_DOWNLOAD_DATASET_HOME=${HF_HOME}/datasets
export HF_ENDPOINT=https://huggingface.co

echo "Model: ${MODEL_NAME}"

echo "CUDA device: ${CUDA_VISIBLE_DEVICES}"

cd train

#--limit=500
python test_task.py \
    --model=${MODEL} \
    --seed=42 \
    --task=${TEST_TASK} \
    --sparse=0.5 \
    --limit=100 \
    --num_shot=0 \
    --do_cr=0 \
    --file_path=${FILE_PATH}
cd ..
