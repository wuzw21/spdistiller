export AMLT_MODE=1

# bash ./run_kd_qat.sh "$@"

export MODEL_NAME=$1
export MODEL_PATH=${AMLT_DATA_DIR}/models/${MODEL_NAME}

export ATTN_SP=$2
export MLP_SP=$2
export W_P=0.0
export DO_CR=$3

export NUM_NODES=1
export NUM_GPUS=4
export MAX_MEMORY="40000MB"
export THRESHOLD_PATH=../threshold/${MODEL_NAME}/${MODEL_NAME}-${ATTN_SP}.txt
export NUM_TRAIN_EPOCHS=4

export TEMPERATURE=0.2

cd train

bash train_amlt.sh \
    ${AMLT_DATA_DIR}/datasets/${MODEL_NAME}/mix_wikitext_alpaca_c4_14000.json \
    ${AMLT_OUTPUT_DIR}/ckpts/${MODEL_NAME}/int4-g64/ \
    ${AMLT_OUTPUT_DIR}/logs/${MODEL_NAME}/int4-g64/ \
    ${NUM_TRAIN_EPOCHS} \
    ${MODEL_PATH} \
    ${MODEL_NAME} \
cd ..