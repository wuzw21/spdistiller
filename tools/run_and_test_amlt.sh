
sparse_values=(0.6 0.7 0.8)
cr_values=(0)
model_name=Meta-Llama-3-8B
sparse_strategy=Static
# test_task=mmlu,wikitext,gsm8k,agieval,arc_easy,arc_challenge,piqa
test_task=wikitext
test_all=1
TASK_MODEL_NAME=static_all_blocks_c4_finetune_no_ste
# TASK_MODEL_NAME=Testall
tasks_array=train
limit=-1
# MODEL_PATH=/mnt/default/projects/wzw-gcrbitdistillerq4/amlt-results/7256335118.30947-8a23b391-3e32-4ca4-a04a-6ab87ed3565d/ckpts/Meta-Llama-3-8B/int4-g64
QUANT=0
USE_LORA=0
if [ "$model_name" = "Meta-Llama-3-8B" ] || [ "$model_name" = "Llama-2-7b-chat-hf" ] || [ "$model_name" = "Llama-2-13b-chat-hf" ]; then
    dataset="c4_processed.json"
    # dataset="mix_alpaca_c4_9000.json"
else
    dataset="mix_wikitext_alpaca_c4_15000.json"
fi
# dataset="mix_wikitext_alpaca_c4_15000.json"

log_params() {
    local log_file="./tools/params_temp.env"  # 日志文件路径
local params=("job_name" "tasks_array" "model_name" "sparse" "do_cr" "sparse_strategy" "test_task" "test_all" "dataset" "limit" "MODEL_PATH" "QUANT" "USE_LORA")  # 参数列表

    # 清空日志文件
    > "$log_file"

    # 遍历参数列表，将每个参数的键值对写入日志文件
    for key in "${params[@]}"; do
        echo "$key=${!key}" >> "$log_file"
    done
}


for sparse in "${sparse_values[@]}"; do
    for do_cr in "${cr_values[@]}"; do
        ori_name=gcrbitdistiller-${model_name}-sparse_${sparse}-cr_${do_cr}
        job_name=${ori_name}_${TASK_MODEL_NAME}
        echo $job_name

        log_params
        
        # amlt map --sla Premium bitdistiller.yaml :gcr_test_task "$job_name" :gcr_bitdistiller --description "lora+$model"
        amlt run --sla Premium bitdistiller.yaml :gcr_bitdistiller "$job_name"
        # amlt run --sla Premium bitdistiller.yaml :gcr_test_task "$job_name" --description "sacle-downstream_task-big_dataset_$model"
        # amlt run --sla Standard mi300.yaml :test_task "$job_name" --description "test_all $model"
    done
done