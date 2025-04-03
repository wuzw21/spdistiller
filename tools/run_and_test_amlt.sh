#!/bin/bash
export AMLT_MODE=1

sparse_values=(0.5)
# sparse_values=(0.7)
cr_values=(0)
# model_name=Llama-2-7b-chat-hf
model_name=Meta-Llama-3-8B
# model_name=Meta-Llama-3-70B-Instruct
# model_name=Mixtral-8x7B-Instruct
sparse_strategy=Static
TEST_TASK=mmlu
TEST_ALL=0
# TASK_MODEL_NAME=test_big_downstreamtask
# TASK_MODEL_NAME=wac8k-cakld-4bit-40G4A100
TASK_MODEL_NAME=q4_distillation-lr_1e-6
# TASK_MODEL_NAME=wac8k-static-STE
for sparse in "${sparse_values[@]}"; do
    for cr in "${cr_values[@]}"; do
        # m1=Llama-2-7b-chat-hf
        ori_name=gcrbitdistiller-${model_name}-sparse_${sparse}-cr_${cr}
        task_name=${ori_name}_${TASK_MODEL_NAME}
        echo $task_name
        # map_name=${task_name}-
        
        # amlt run --sla standard mi300.yaml :gcrbitdistiller "$task_name" --extra-args "$model_name $sparse $cr $sparse_strategy $TEST_TASK $TEST_ALL"

        amlt run --sla premium bitdistiller.yaml :gcrbitdistiller "$task_name" --extra-args "$model_name $sparse $cr $sparse_strategy $TEST_TASK $TEST_ALL"

        # amlt run --sla premium bitdistiller.yaml :gcrbitdistiller "$task_name" --extra-args "$model_name $sparse $cr $sparse_strategy $TEST_TASK $TEST_ALL"

        # amlt run --sla premium bitdistiller.yaml :gcr_generate_threshold "$task_name" --extra-args "$model_name"
        
        # amlt map --sla Premium bitdistiller.yaml :gcr_bd_3_test_task "$task_name" :gcrbitdistiller --description "sacle-downstream_task-big_dataset_$model" --extra-args "$model_name $sparse $cr $sparse_strategy $TEST_TASK $TEST_ALL"
        # amlt map --sla standard mi300.yaml :gcr_bd_test_task "$task_name" :gcrbitdistiller --description "downstream_task-big_dataset_$model" --extra-args "$model_name $sparse $cr $sparse_strategy $TEST_TASK $TEST_ALL"

        # amlt run --sla Premium bitdistiller.yaml :gcr_bd_test_task "$task_name" --extra-args "$model_name $sparse $cr $sparse_strategy $TEST_TASK $TEST_ALL"
        # amlt run --sla standard mi300.yaml :gcr_test_task "$task_name" --extra-args "$model_name $sparse $cr $sparse_strategy $TEST_TASK $TEST_ALL"
    done
done