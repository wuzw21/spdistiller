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

SKU=${SKU:-"40G4-A100"}
echo "SKU: $SKU"
export NUM_GPUS=$(echo "$SKU" | sed -E 's/.*G([0-9]+)-.*/\1/')
echo "NUM_GPUS: $NUM_GPUS"

echo $MODEL_NAME
echo $MODEL_PATH

cd data/generation

JSON_PATH1="${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/wikitext_T0.2_N1024_S42_5000.json"
JSON_PATH2="${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/alpaca_T0.2_N1024_S42_5000.json"
JSON_PATH3="${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/c4_T0.2_N1024_S42_5000.json"

OUTPUT_JSON="${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/mix_wikitext_alpaca_c4_15000.json"

python mix_data.py --output "$OUTPUT_JSON" --inputs "$JSON_PATH1" "$JSON_PATH2" "$JSON_PATH3"

cd ../..

