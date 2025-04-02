#!/bin/bash

sparse_values=(0.5 0.6 0.7 0.8)
cr_values=(0)
model_name=Meta-Llama-3-8B
sparse_strategy=Static
test_task=mmlu
test_all=0
# TASK_MODEL_NAME=wac8k-cakld-4bit-80G4A100
TASK_MODEL_NAME=q4-distill
tasks_array=train

if [ "$model_name" = "Meta-Llama-3-8B" ] || [ "$model_name" = "Llama-2-7b-chat-hf" ] || [ "$model_name" = "Llama-2-13b-chat-hf" ]; then
    dataset="mix_alpaca_c4_9000.json"
else
    dataset="mix_wikitext_alpaca_c4_15000.json"
fi
# dataset="mix_wikitext_alpaca_c4_15000.json"

log_params() {
    local log_file="./tools/params_temp.env"  # 日志文件路径
    local params=("job_name" "tasks_array" "model_name" "sparse" "do_cr" "sparse_strategy" "test_task" "test_all" "dataset")  # 参数列表

    # 清空日志文件
    > "$log_file"

    # 遍历参数列表，将每个参数的键值对写入日志文件
    for key in "${params[@]}"; do
        echo "$key=${!key}" >> "$log_file"
    done
}


for sparse in "${sparse_values[@]}"; do
    for do_cr in "${cr_values[@]}"; do
        ori_name=${model_name}-sparse_${sparse}-cr_${do_cr}
        job_name=${ori_name}_${TASK_MODEL_NAME}
        echo $job_name

        log_params
        
        # amlt map --sla Premium bitdistiller.yaml :gcr_bd_test_task "$job_name" :gcrbitdistiller --description "big_dataset_$model"
        amlt run --sla Premium bitdistiller.yaml :gcr_bitdistiller "$job_name" --description "sacle-downstream_task-big_dataset_$model"
    done
done