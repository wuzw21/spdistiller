#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.3


export MODEL_NAME=Llama-2-13b-chat-hf
# export MODEL_NAME=Llama-2-7b-chat-hf
# export MODEL_NAME=Qwen2.5-0.5B-Instruct
# export MODEL_NAME=Mixtral-8x7B-Instruct
# export MODEL=/data/wzw/models/Llama/${MODEL_NAME}


export TEST_TASK=wiki

export HF_HOME=${HOME}/Downloads/huggingface
export HF_DOWNLOAD_DATASET_HOME=${HF_HOME}/datasets
export HF_ENDPOINT=https://huggingface.co

export MODEL=/data/wzw/models/${MODEL_NAME}

echo "Model: ${MODEL_NAME}"

echo "CUDA device: ${CUDA_VISIBLE_DEVICES}"

echo "Model: ${MODEL}"

cd train

#--limit=500
python threshold_generation.py \
    --model=${MODEL} \
    --seed=42 \
    --task=${TEST_TASK} \
    --limit=100 \
    --num_shot=0 \
    --file_path=/data/wzw/Projects/wzw/BitDistiller-Q4_0/threshold/${MODEL_NAME}-all \
    --sparse=0 \
    --do_generation \
    --save_activations \
    --test_all=1
cd ..
