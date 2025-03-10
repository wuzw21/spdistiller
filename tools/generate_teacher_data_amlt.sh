#!/biin/bash

export OMP_NUM_THREADS=16

export MODEL_NAME=Llama-2-7b-chat-hf
export MODEL_PATH=${AMLT_DATA_DIR}/models/${MODEL_NAME}

SKU=$SKU
echo "SKU: $SKU"
export NUM_GPUS=$(echo "$SKU" | sed -E 's/.*G([0-9]+)-.*/\1/')
echo "NUM_GPUS: $NUM_GPUS"

echo $MODEL_NAME
echo $MODEL_PATH

export ENABLE_PREDICTOR=0
export ENABLE_SPARSE_INFER=0
export ENABLE_TENSOR_SAVER=0
cd data/generation

# wikitext, 5000
bash generate.sh ${MODEL_PATH} wikitext ${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME} 16 5000 ${NUM_GPUS}

# # alpaca, 5000
# bash generate.sh ${MODEL_PATH} alpaca ${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/ 16 5000 ${NUM_GPUS}

# # c4, 5000
# bash generate.sh ${MODEL_PATH} c4 ${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/ 16 5000 ${NUM_GPUS}

# JSON_PATH1="${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/wikitext_T0.2_N1024_S42_5000.json"
# JSON_PATH2="${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/alpaca_T0.2_N1024_S42_5000.json"
# JSON_PATH3="${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/c4_T0.2_N1024_S42_5000.json"

# OUTPUT_JSON="${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}/mix_wikitext_alpaca_c4_15000.json"

# python mix_data.py --output "$OUTPUT_JSON" --inputs "$JSON_PATH1" "$JSON_PATH2" "$JSON_PATH3"

cd ../..

