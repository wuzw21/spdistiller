SKU=${SKU:-"40G4-A100"}

export NUM_GPUS=$(echo "$SKU" | sed -E 's/.*G([0-9]+)-.*/\1/')

echo "NUM_GPUS: $NUM_GPUS"

export MAX_MEMORY=$(echo "$SKU" | sed -E 's/^([0-9]+)G.*/\1/' | awk '{print $1 * 10000}')

echo "MAX_MEMORY: $MAX_MEMORY MB"


# export MODEL_PATH=${AMLT_DATA_DIR}/models/Llama-2-7b-chat-hf
# export MODEL_NAME=Llama-2-7b-chat-hf
# export MODEL_PATH=${AMLT_DATA_DIR}/models/Llama-2-13b-chat-hf
# export MODEL_NAME=Llama-2-13b-chat-hf
#export MODEL_PATH=${AMLT_DATA_DIR}/models/Meta-Llama-3-8B-Instruct
#export MODEL_NAME=Meta-Llama-3-8B-Instruct
export MODEL_PATH=${AMLT_DATA_DIR}/models/Meta-Llama-3.1-8B-Instruct
export MODEL_NAME=Meta-Llama-3.1-8B-Instruct

#export TEMPERATURE=0.7
export TEMPERATURE=0.2

export NUM_TRAIN_EPOCHS=4

export NUM_NODES=1

export ATTN_SP=0.9
export MLP_SP=0.9
export W_P=0.0
export DO_CR=1

cd train

# c4_T${TEMPERATURE}_N1024_S42_4096.json
bash train_amlt.sh \
    ${AMLT_DATA_DIR}/datasets/${MODEL_NAME}/mix_wikitext_alpaca_c4_8000.json \
    ${AMLT_OUTPUT_DIR}/ckpts/${MODEL_NAME}/int4-g64/ \
    ${AMLT_OUTPUT_DIR}/logs/${MODEL_NAME}/int4-g64/ \
    ${NUM_TRAIN_EPOCHS} \
    ${MODEL_PATH} \
    ${MODEL_NAME} \
    ${NUM_NODES} \
    ${NUM_GPUS} \
    ${MAX_MEMORY}

cd ..
