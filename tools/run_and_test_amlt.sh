#!/bin/bash
export AMLT_MODE=1

sparse_values=(0.5 0.6 0.7 0.8 0.9)
cr_values=(0 1)
model_name=Meta-Llama-3-8B
# sparse_values=(0.5)
# cr_values=(0)
for sparse in "${sparse_values[@]}"; do
    for cr in "${cr_values[@]}"; do
        task_name="gcrbitdistiller-${model_name}-sparse_${sparse}-cr_${cr}_wac8k"
        
        amlt run --sla Premium bitdistiller.yaml :gcrbitdistiller "$task_name" --extra-args "$model_name $sparse $cr"
        
        # amlt map --sla Premium bitdistiller.yaml :gcrbitdistiller_test_task "$task_name" --description "mmlu-and-wiki-$model" --extra-args "$model_name $sparse $cr"
    done
done