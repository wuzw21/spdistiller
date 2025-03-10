export AMLT_MODE=1

# bash ./run_kd_qat.sh "$@"

export MODEL_NAME=$1
export MODEL_PATH=${AMLT_DATA_DIR}/models/${MODEL_NAME}

export ATTN_SP=$2
export MLP_SP=$2
export W_P=0.0
export DO_CR=$3

export NUM_NODES=1


if [ -z "$SKU" ]; then
    NUM_GPUS=4
    MAX_MEMORY="40000MB"
else
    NUM_GPUS=$(echo "$SKU" | awk -F'G|-' '{print $2}')
    MAX_MEMORY_GB=$(echo "$SKU" | grep -o '^[0-9]\+')
    MAX_MEMORY=$((MAX_MEMORY_GB * 1000))MB
fi
export NUM_GPUS
export MAX_MEMORY

echo $SKU
echo $NUM_GPUS
echo $MAX_MEMORY

SPARSE_STRATEGY=$4
if [ "$SPARSE_STRATEGY" = "Static" ]; then
    export THRESHOLD_PATH="../threshold/${MODEL_NAME}/${MODEL_NAME}-${ATTN_SP}.txt"
else
    export THRESHOLD_PATH="zwwz"
fi

export NUM_TRAIN_EPOCHS=5

export TEMPERATURE=0.2

cd train

# mix_wikitext_alpaca_c4_15000.json
# mix_alpaca_c4_9000

bash train_amlt.sh \
    ${AMLT_DATA_DIR}/datasets/${MODEL_NAME}/mix_wikitext_alpaca_c4_15000.json \
    ${AMLT_OUTPUT_DIR}/ckpts/${MODEL_NAME}/int4-g64/ \
    ${AMLT_OUTPUT_DIR}/logs/${MODEL_NAME}/int4-g64/ \
    ${NUM_TRAIN_EPOCHS} \
    ${MODEL_PATH} \
    ${MODEL_NAME} \

cd ..

bash tools/run_test_task_amlt.sh $1 $2 $3 $4 ${AMLT_OUTPUT_DIR}/ckpts/${MODEL_NAME}/int4-g64
