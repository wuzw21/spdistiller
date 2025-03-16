#!/biin/bash

export OMP_NUM_THREADS=16

export HF_HOME=/data/wzw
export DATASET_DIR=data/datasets
export MODEL_NAME=Mixtral-8x7B-Instruct
export MODEL_PATH=/data/wzw/models/${MODEL_NAME}
export SPARSITY_PATH=/home/donglinbai/Projects/wzw/BitDistiller-Q4_0/threshold/llama-3-0.7.txt


cd data/generation

# wikitext, 5000
bash generate.sh ${MODEL_PATH} wikitext ../datasets/${MODEL_NAME} 16 5000

# alpaca, 5000
bash generate.sh ${MODEL_PATH} alpaca ../datasets/${MODEL_NAME}/ 16 5000

# c4, 4096
bash generate.sh ${MODEL_PATH} c4 ../datasets/${MODEL_NAME}/ 16 4096

python mix_data.py

cd ../..
