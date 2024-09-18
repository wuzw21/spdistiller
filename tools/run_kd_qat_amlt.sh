
#export MODEL_PATH=NousResearch/Meta-Llama-3-8B
#export MODEL_NAME=Meta-Llama-3-8B
export MODEL_PATH=${AMLT_DATA_DIR}/models/Meta-Llama-3-8B
export MODEL_NAME=Meta-Llama-3-8B
#export MODEL_PATH=NousResearch/Llama-2-13b-chat-hf
#export MODEL_NAME=Llama-2-13b-chat-hf

#export TEMPERATURE=0.7
export TEMPERATURE=0.2

export NUM_TRAIN_EPOCHS=4

export NUM_GPUS=8

cd train

# c4_T${TEMPERATURE}_N1024_S42_4096.json
bash train_amlt.sh \
    ${AMLT_DATA_DIR}/datasets/${MODEL_NAME}/mix_alpaca_c4_9000.json \
    ${AMLT_OUTPUT_DIR}/ckpts/${MODEL_NAME}/int4-g64/ \
    ${AMLT_OUTPUT_DIR}/logs/${MODEL_NAME}/int4-g64/ \
    ${NUM_TRAIN_EPOCHS} \
    ${MODEL_PATH} \
    ${MODEL_NAME} \
    ${NUM_GPUS}

cd ..
