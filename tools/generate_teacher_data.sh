#!/biin/bash

export OMP_NUM_THREADS=16

export HF_HOME=/data/fuchengjia/Downloads/huggingface

export MODEL_NAME=$1
export MODEL_PATH=${HF_HOME}/models/${MODEL_NAME}

export ENABLE_PREDICTOR=0

cd data/generation

# wikitext, 3000
#bash generate.sh ${MODEL_PATH} wikitext ../datasets/${MODEL_NAME} 16 3000

# alpaca, 5000
#bash generate.sh ${MODEL_PATH} alpaca ../datasets/${MODEL_NAME}/ 16 5000

# c4, 8000
#bash generate.sh ${MODEL_PATH} c4 ../datasets/${MODEL_NAME}/ 16 4096

python mix_data.py

cd ../..
