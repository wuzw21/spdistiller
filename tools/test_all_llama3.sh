#!/bin/bash

# TODO : fix all Models
MODELS=(
    projects/wzw-gcrbitdistillerq4/amlt-results/7258901328.44367-290c702f-5b09-4e96-ae5f-bc54391188bb
)
SPARSES=(
    0.7
)
TEST_ALL=(
    1
)
Llama-3-cakld
0.7 cakld(TEST_ALL): projects/wzw-gcrbitdistillerq4/amlt-results/7258901328.44367-290c702f-5b09-4e96-ae5f-bc54391188bb
0.6 cakld: projects/wzw-gcrbitdistillerq4/amlt-results/7258976559.08562-6f8c9bcd-2e63-4bc6-84c3-839c23ee32fa
0.5 cakld: projects/wzw-gcrbitdistillerq4/amlt-results/7258976620.02019-49367bd4-424f-49bd-89e8-cebf80488af6

LLama-3-Ste
0.8(TEST_ALL): projects/wzw-gcrbitdistillerq4/amlt-results/7260183513.51709-a9f15020-08d9-4430-8043-efd351b3a4c6
0.7: projects/wzw-gcrbitdistillerq4/amlt-results/7259969968.57103-c6b48a3d-572b-4c88-9c45-812c51ffa78d
0.6: projects/wzw-gcrbitdistillerq4/amlt-results/7260183546.70913-96d8561c-472d-4552-9391-0a413b1c396f
0.5: projects/wzw-gcrbitdistillerq4/amlt-results/7260183682.56628-648b8392-337c-4722-96d7-257108009ec8

FULL_MODEL_PATH=/mnt/default/${MODEL_PATH}/ckpts/Meta-Llama-3-8B/int4-g64


export MODEL_NAME="Meta-Llama-3-8B"
export TEST_TASK="mmlu"

export HF_HOME="${HOME}/Downloads/huggingface"
export HF_DOWNLOAD_DATASET_HOME="${HF_HOME}/datasets"
export HF_ENDPOINT="https://huggingface.co"

export SPARSE_STRATEGY='Static'

# 使用 bash 数组索引迭代
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    SPARSE="${SPARSES[$i]}"
    DO_CR=0
    export AMLT_MAP_INPUT_DIR=${MODEL}
    bash tools/run_test_task_amlt.sh "${MODEL_NAME}" "${SPARSE}" "${DO_CR}" "${SPARSE_STRATEGY}" $TEST_TASK $TEST_ALL
done