#!/bin/bash
# 
source tools/params_temp.env

export CURRENT_DIR=$(dirname $(dirname $(realpath "$0")))
# change this to your model directory
export MODEL_DIR=/data/wzw/models/${MODEL_NAME}
export OUTPUT_DIR=/data/wzw/Projects/bitdistiller
export DATA_DIR=$CURRENT_DIR/data

export MODEL_NAME=$model_name
export SPARSE=$sparse
export DO_CR=$do_cr
export SPARSE_STRATEGY=$sparse_strategy
export TEST_TASK=$test_task
export TEST_ALL=$test_all
export DATASET=$dataset
export MODEL_PATH=${MODEL_DIR}/${MODEL_NAME}
export THRESHOLD_PATH="${CURRENT_DIR}/data/threshold/${MODEL_NAME}/sparse-${SPARSE}.json"

### =================================================================================== ###

export EASY_TEST=1

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
    fi
done