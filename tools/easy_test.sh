#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.3


# export MODEL_NAME=Llama-2-7b-chat-hf
# export MODEL_NAME=Meta-Llama-3-8B
export MODEL_NAME=Mixtral-8x7B-Instruct
export SPARSE=0.5
export DO_CR=0
export SPARSE_STRATEGY=Static
export TEST_TASK=wiki
export TEST_ALL=1

bash tools/run_test_task.sh $MODEL_NAME $SPARSE $DO_CR $SPARSE_STRATEGY $TEST_TASK $TEST_ALL