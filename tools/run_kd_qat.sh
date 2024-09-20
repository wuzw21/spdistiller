

export PROJECT_DATA_HOME=/data/fuchengjia2/Projects/BitDistiller/data

export HF_HOME=/data/fuchengjia2/Downloads/huggingface
export MODEL_NAME=Llama-2-7b-chat-hf
export MODEL_PATH=${HF_HOME}/models/${MODEL_NAME}

export NUM_TRAIN_EPOCHS=4

export NUM_GPUS=4

cd train

bash train.sh \
    ${PROJECT_DATA_HOME}/datasets/${MODEL_NAME}/mix_alpaca_c4_9000.json \
    ${PROJECT_DATA_HOME}/ckpts/${MODEL_NAME}/int4-g64/ \
    ${PROJECT_DATA_HOME}/logs/${MODEL_NAME}/int4-g64/ \
    ${NUM_TRAIN_EPOCHS} \
    ${MODEL_PATH} \
    ${MODEL_NAME} \
    ${NUM_GPUS}

cd ..
