#!/biin/bash

export OMP_NUM_THREADS=16

#export MODEL_NAME=Llama-2-7b-hf
#export MODEL_PATH=${AMLT_DATA_DIR}/models/Llama-2-7b-hf
export MODEL_NAME=Llama-2-7b-chat-hf
export MODEL_PATH=${AMLT_DATA_DIR}/models/Llama-2-7b-chat-hf
#export MODEL_NAME=Llama-2-13b-chat-hf
#export MODEL_PATH=${AMLT_DATA_DIR}/models/Llama-2-13b-chat-hf
#export MODEL_NAME=Llama-2-70b-chat-hf
#export MODEL_PATH=${AMLT_DATA_DIR}/models/Llama-2-70b-chat-hf
#export MODEL_NAME=Mixtral-8x7B-v0.1-c4-ft
#export MODEL_PATH=${AMLT_DATA_DIR}/models/Mixtral-8x7B-v0.1-c4-ft
#export MODEL_NAME=Meta-Llama-3-8B-Instruct
#export MODEL_PATH=${AMLT_DATA_DIR}/models/Meta-Llama-3-8B-Instruct
#export MODEL_NAME=Meta-Llama-3-70B-Instruct
#export MODEL_PATH=${AMLT_DATA_DIR}/models/Meta-Llama-3-70B-Instruct
#export MODEL_NAME=Meta-Llama-3.1-8B-Instruct
#export MODEL_PATH=${AMLT_DATA_DIR}/models/Meta-Llama-3.1-8B-Instruct
#export MODEL_NAME=Meta-Llama-3.1-70B-Instruct
#export MODEL_PATH=${AMLT_DATA_DIR}/models/Meta-Llama-3.1-70B-Instruct

export ENABLE_PREDICTOR=0
export ENABLE_SPARSE_INFER=0
export ENABLE_TENSOR_SAVER=0

cd data/generation

# wikitext, 3000
bash generate.sh ${MODEL_PATH} wikitext ${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME} 16 3000

# alpaca, 5000
bash generate.sh ${MODEL_PATH} alpaca ${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/ 16 5000

# c4, 8000
bash generate.sh ${MODEL_PATH} c4 ${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/ 16 4096

python mix_data.py

cd ../..
