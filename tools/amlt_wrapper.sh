#!/bin/bash

source tools/params_temp.env

export CURRENT_DIR=$(dirname $(dirname $(realpath "$0")))

export OUTPUT_DIR=${AMLT_OUTPUT_DIR}
export DATA_DIR=${AMLT_DATA_DIR}

export MODEL_NAME=$model_name
export SPARSE=$sparse
export DO_CR=$do_cr
export SPARSE_STRATEGY=$sparse_strategy
export TEST_TASK=$test_task
export TEST_ALL=$test_all
export DATASET=$dataset

if [ -n "${AMLT_MAP_INPUT_DIR}" ]; then
    export MODEL_DIR=${AMLT_MAP_INPUT_DIR}/ckpts/${MODEL_NAME}/int4-g64
else
    export MODEL_DIR=${AMLT_DATA_DIR}/models/${MODEL_NAME}
fi
echo "MODEL_DIR: ${MODEL_DIR}"
export MODEL_PATH=${MODEL_DIR}
# export MODEL_PATH=${MODEL_DIR}/${MODEL_NAME}
export THRESHOLD_PATH="${CURRENT_DIR}/data/threshold/${MODEL_NAME}/sparse-${SPARSE}.json"

### =================================================================================== ###

tasks_array=$tasks_array

IFS=',' read -ra params_array <<< "$tasks_array"

echo params_array: ${tasks_array[@]}

for task in "${params_array[@]}"; do
    echo current_task: $task
    # continue
    if [[ $task =~ "test" ]]; then
        
        bash tools/test_task.sh
    elif [[ $task =~ "generate_teacher_data" ]]; then
        bash tools/generate_teacher_data.sh
    elif [[ $task =~ "generate_threshold" ]]; then
        bash tools/generate_threshold.sh
    elif [[ $task =~ "chat" ]]; then
        bash tools/chat.sh
    elif [[ $task =~ "train" ]]; then
        bash tools/train.sh
        export MODEL_DIR=${OUTPUT_DIR}/ckpts
        export MODEL_PATH=${MODEL_DIR}/${MODEL_NAME}
        bash tools/test_task.sh
    fi
done