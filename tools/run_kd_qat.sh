export DATA_DIR=/home/donglinbai/Projects/wzw/BitDistiller-Q4_0/data
export OUTPUT_DIR=/data/wzw/models

if [ "${AMLT_MODE:-0}" -eq 1 ]; then
    export DATA_PATH=${AMLT_DATA_DIR}
    export OUTPUT_DIR=${AMLT_OUTPUT_DIR}
fi

export MODEL_DIR=/data/wzw/models
export MODEL_NAME=$1
export MODEL_PATH=${MODEL_DIR}/${MODEL_NAME}

export ATTN_SP=$2
export MLP_SP=$2
export W_P=0.0
export DO_CR=$3

export NUM_NODES=1
export NUM_GPUS=4
export MAX_MEMORY="24000MB"
export THRESHOLD_PATH=../threshold/${MODEL_NAME}/${MODEL_NAME}-${ATTN_SP}.txt
export NUM_TRAIN_EPOCHS=4

export TEMPERATURE=0.2

cd train

bash train.sh \
    ${DATA_DIR}/datasets/${MODEL_NAME}/mix_alpaca_c4_9000.json \
    ${OUTPUT_DIR}/ckpts/${MODEL_NAME}/int4-g64/ \
    ${OUTPUT_DIR}/logs/${MODEL_NAME}/int4-g64/ \
    ${NUM_TRAIN_EPOCHS} \
    ${MODEL_PATH} \
    ${MODEL_NAME}

cd ..
