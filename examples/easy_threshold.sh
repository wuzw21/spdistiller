#!/bin/bash

# replace with your path
export DATA_DIR=$(dirname $(dirname $(realpath "$0")))
export MODEL_DIR=/data/wzw/models
export OUTPUT_DIR=/data/wzw/Projects

# replace with your params
export MODEL_NAME=Qwen2.5-0.5B-Instruct
export SPARSE=0.5
export DO_CR=0
export SPARSE_STRATEGY=Static
export TEST_TASK=wiki
export TEST_ALL=1
# set EASY_TEST=1 to use easy-test-debug mode
export EASY_TEST=1

export MODEL_PATH=${MODEL_DIR}/${MODEL_NAME}
export THRESHOLD_PATH="${DATA_DIR}/data/threshold/${MODEL_NAME}/sparse-${SPARSE}.json"

echo "Model: ${MODEL_NAME}"
echo "MODEL path set to: $MODEL_PATH"
echo "THRESHOLD path set to: $THRESHOLD_PATH"

# run scripts
bash tools/generate_threshold.sh