#!/bin/bash
export AMLT_MODE=1

# export MODEL_NAME=Llama-2-7b-chat-hf
export MODEL_NAME=$1
export MODEL_PATH=${AMLT_DATA_DIR}/models/${MODEL_NAME}
# export MODEL=/data/wzw/models/Llama/${MODEL_NAME}



export HF_HOME=${HOME}/Downloads/huggingface
export HF_DOWNLOAD_DATASET_HOME=${HF_HOME}/datasets
export HF_ENDPOINT=https://huggingface.co

echo "Model: ${MODEL_NAME}"

echo "CUDA device: ${CUDA_VISIBLE_DEVICES}"

cd train

#--limit=500
python threshold_generation.py \
    --model=${MODEL_PATH} \
    --seed=42 \
    --task=${TEST_TASK} \
    --limit=100 \
    --num_shot=0 \
    --file_path=${AMLT_OUTPUT_DIR}/${MODEL_NAME} \
    --sparse=0 \
    --do_generation \
    --save_activations \
    --test_all=1
cd ..
