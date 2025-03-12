#!/bin/bash
export AMLT_MODE=0
export SKU=24G4-4090

sparse_values=(0.5)
# cr_values=(0 1)
sparse_values=(0)
cr_values=(0)
model_name=Meta-Llama-3-8B
# sparse_values=(0.5)
# cr_values=(0)
for sparse in "${sparse_values[@]}"; do
    for cr in "${cr_values[@]}"; do
        task_name="gcrbitdistiller-${model_name}-sparse_${sparse}-cr_${cr}_wac8k"
        
        bash ./tools/run_kd_qat.sh $model_name $sparse $cr
        
        # bash ./tools/run_test_task.sh $model_name $sparse $cr
    done
done