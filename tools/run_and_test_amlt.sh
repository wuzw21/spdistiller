#!/bin/bash
export AMLT_MODE=1

sparse_values=(0.5 0.6 0.7 0.8)
# sparse_values=(0.7)
cr_values=(0)
model_name=Llama-2-7B-chat-hf
# model_name=Meta-Llama-3-8B
for sparse in "${sparse_values[@]}"; do
    for cr in "${cr_values[@]}"; do
        # m1=Llama-2-7b-chat-hf
        task_name="gcrbitdistiller-${model_name}-sparse_${sparse}-cr_${cr}_wac8k-static-STE"
        
        amlt run --sla Premium bitdistiller.yaml :gcrbitdistiller "$task_name" --extra-args "$model_name $sparse $cr"
        
        # amlt map --sla Premium bitdistiller.yaml :gcrbitdistiller_test_task "$task_name" --description "mmlu-and-wiki-$model" --extra-args "$model_name $sparse $cr"

        # amlt run --sla Premium bitdistiller.yaml :gcrbitdistiller_test_task "$task_name" --extra-args "$model_name $sparse $cr"
    done
done