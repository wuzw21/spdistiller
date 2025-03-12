#!/bin/bash
export AMLT_MODE=1

sparse_values=(0.8)
# sparse_values=(0.7)
cr_values=(0)
# model_name=Llama-2-7b-chat-hf
# model_name=Meta-Llama-3-8B
# model_name=Meta-Llama-3-70B-Instruct
model_name=Mixtral-8x7B-Instruct
sparse_strategy=Static
TEST_TASK=mmlu
TEST_ALL=1
for sparse in "${sparse_values[@]}"; do
    for cr in "${cr_values[@]}"; do
        # m1=Llama-2-7b-chat-hf
        task_name="gcrbitdistiller-${model_name}-sparse_${sparse}-cr_${cr}_test_task"
        
        # amlt run --sla standard mi300.yaml :gcrbitdistiller "$task_name" --extra-args "$model_name $sparse $cr $sparse_strategy $TEST_TASK $TEST_ALL"

        # amlt run --sla premium bitdistiller.yaml :gcr_generate_threshold "$task_name" --extra-args "$model_name"
        
        # amlt map --sla Premium bitdistiller.yaml :gcr_test_task "$task_name" --description "mmlu-and-wiki-$model" --extra-args "$model_name $sparse $cr $sparse_strategy $TEST_TASK $TEST_ALL"

        amlt run --sla Premium bitdistiller.yaml :gcr_test_task "$task_name" --extra-args "$model_name $sparse $cr $sparse_strategy $TEST_TASK $TEST_ALL"
    done
done