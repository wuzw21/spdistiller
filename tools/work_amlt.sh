#!/biin/bash

export OMP_NUM_THREADS=16

# export MODEL_NAME=Llama-2-7b-hf
# export MODEL_PATH=${AMLT_DATA_DIR}/models/Llama-2-7b-hf
# export MODEL_NAME=Llama-2-7b-chat-hf
# export MODEL_PATH=${AMLT_DATA_DIR}/models/Llama-2-7b-chat-hf
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
export MODEL_NAME=Meta-Llama-3.1-8B-Instruct
export MODEL_PATH=${AMLT_DATA_DIR}/models/Meta-Llama-3.1-8B-Instruct
# export MODEL_NAME=Meta-Llama-3.1-70B-Instruct
# export MODEL_PATH=${AMLT_DATA_DIR}/models/Meta-Llama-3.1-70B-Instruct


OUTPUT_JSON="${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/mix_wikitext_alpaca_c4_8000.json"



cd ../..

