#!/biin/bash

export OMP_NUM_THREADS=16

export HF_HOME=/data/wzw
export DATASET_DIR=data/datasets
export MODEL_NAME=$1
export MODEL_PATH=${HF_HOME}/models/${MODEL_NAME}

export ENABLE_PREDICTOR=0

export ENABLE_PREDICTOR=0
export ENABLE_SPARSE_INFER=0
export ENABLE_TENSOR_SAVER=0

cd data/generation

# wikitext, 5000
bash generate.sh ${MODEL_PATH} wikitext ../datasets/${MODEL_NAME} 16 5000

# alpaca, 5000
bash generate.sh ${MODEL_PATH} alpaca ../datasets/${MODEL_NAME}/ 16 5000

# c4, 4096
bash generate.sh ${MODEL_PATH} c4 ../datasets/${MODEL_NAME}/ 16 4096

python mix_data.py

cd ../..
