#!/bin/bash
 
source tools/params_temp.env
echo "====== Environment Variables ======="
cat tools/params_temp.env
### ================================================================================= ###
# set path environment variables
export CURRENT_DIR=$(dirname $(dirname $(realpath "$0")))
# change this to your model directory
export MODEL_DIR=${MODEL_DIR:-"/data/wzw/models"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/data/wzw/Projects/bitdistiller"}
export DATA_DIR=${DATA_DIR:-$CURRENT_DIR/data}
echo "MODEL_DIR: $MODEL_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "DATA_DIR: $DATA_DIR"
### ================================================================================= ###
# load environment variables
export MODEL_NAME=$model_name
export SPARSE=$sparse
export DO_CR=$do_cr
export SPARSE_STRATEGY=$sparse_strategy
export TEST_TASK=$test_task
export TEST_ALL=$test_all
export DATASET=$dataset
export LIMIT=$limit
export MODEL_PATH=${MODEL_PATH:-${MODEL_DIR}/${MODEL_NAME}}
export THRESHOLD_PATH=${THRESHOLD:-"${DATA_DIR}/threshold/${MODEL_NAME}/sparse-${SPARSE}.json"}
export QUANT=${QUANT:-0}
export USE_LORA=${USE_LORA:-0}
export LORA_CHECKPOINT=${LORA_CHECKPOINT:-${MODEL_PATH}}
### =================================================================================== ###
# load debug environment variables
export DEBUG_CROSSLAYER=${DEBUG_CROSSLAYER:-0}
export EASY_TEST=1
### =================================================================================== ###

# execute all tasks
tasks_array=$tasks_array
echo "====== tasks ======="
IFS=',' read -ra params_array <<< "$tasks_array"

echo params_array: ${tasks_array[@]}

for task in "${params_array[@]}"; do
    echo "====== current_task $task ======="
    # continue
    if [[ $task =~ "test" ]]; then
        bash tools/test_task.sh
    elif [[ $task == "test_all" ]]; then
        bash tools/test_task_all.sh
    elif [[ $task =~ "generate_teacher_data" ]]; then
        bash tools/generate_teacher_data.sh
    elif [[ $task =~ "generate_threshold" ]]; then
        bash tools/generate_threshold.sh
    elif [[ $task =~ "chat" ]]; then
        bash tools/chat.sh
    elif [[ $task =~ "train" ]]; then
        bash tools/train.sh
        export MODEL_DIR=${OUTPUT_DIR}/ckpts
        if USE_LORA=1; then
            export LORA_CHECKPOINT=${MODEL_DIR}/${MODEL_NAME}
        else
            export MODEL_PATH=${MODEL_DIR}/${MODEL_NAME}
        fi
        bash tools/test_task.sh
    fi
done