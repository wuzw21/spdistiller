#!/bin/bash

export OMP_NUM_THREADS=16

export MODEL_NAME=Qwen2.5-0.5B-Instruct
export MODEL_PATH=/data/wzw/models/${MODEL_NAME}
export SPARSE=0.7
temperature=0.7
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

datasets=("wikitext" "alpaca" "c4")
max_sample=4
batch_size=16

cd data/generation

output_path=/home/donglinbai/Projects/wzw/BitDistiller-Q4_0/data/datasets/${MODEL_NAME}

# for dataset in "${datasets[@]}"; do
#     torchrun --nproc_per_node=$NUM_GPUS generate.py \
#         --base_model ${MODEL_PATH} \
#         --dataset_name $dataset \
#         --out_path  output_path\
#         --batch_size $batch_size \
#         --max_sample $max_sample \
#         --threshold_path ../../threshold/${MODEL_NAME}/sparse-${SPARSE}.json
# done

json_paths=()
for dataset in "${datasets[@]}"; do
    json_paths+=("${output_path}/${dataset}_T${temperature}_N1024_S42_${max_sample}.json")
done

# 合并 JSON 文件
OUTPUT_JSON="${output_path}/mix_wikitext_alpaca_c4_15000.json"
python mix_data.py --output "$OUTPUT_JSON" --inputs "${json_paths[@]}"

cd ../..

