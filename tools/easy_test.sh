#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.3


export MODEL_NAME=Llama-2-7B-chat-hf
export MODEL=/data/wzw/models/Llama/${MODEL_NAME}
# export MODEL=/data/wzw/models/Meta-Llama-3-8B
# export MODEL=/data/wzw/models/Meta-Llama-3-8B-0.8-STE/ckpts/Meta-Llama-3-8B/int4-g64

export TEST_TASK=wiki

export ENABLE_PREDICTOR=1
export ENABLE_SPARSE_INFER=1
export ENABLE_TENSOR_SAVER=0

export HF_HOME=${HOME}/Downloads/huggingface
export HF_DOWNLOAD_DATASET_HOME=${HF_HOME}/datasets
export HF_ENDPOINT=https://huggingface.co


export ATTN_SP=0.8
export MLP_SP=0.8
export W_P=0
export DO_CR=1
export DEBUG_CROSSLAYER=0
SPARSE_STRATEGY=Static
# export THRESHOLD_PATH=/home/donglinbai/Projects/wzw/TEAL/data/threshold/Meta-Llama-3-8B-0.7.txt
# export THRESHOLD_PATH=zwwz
export THRESHOLD_PATH=/home/donglinbai/Projects/wzw/BitDistiller-Q4_0/threshold/Meta-Llama-3-8B/Meta-Llama-3-8B-${ATTN_SP}.txt
export ACTIVATE_LAYER=0

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
    --limit=20 \
    --num_shot=0 \
    --do_cr=${DO_CR} \
    --file_path=${THRESHOLD_PATH} \
    --sparse_strategy=${SPARSE_STRATEGY} 
    # --test_all 1
cd ..
