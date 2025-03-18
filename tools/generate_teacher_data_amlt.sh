#!/biin/bash

export OMP_NUM_THREADS=16

export MODEL_NAME=Mixtral-8x7B-Instruct
export MODEL_PATH=${AMLT_DATA_DIR}/models/${MODEL_NAME}
export SPARSE=0.7

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

datasets=("wikitext" "alpaca" "c4")
max_sample=5000
batch_size=16

cd data/generation

output_path=${AMLT_OUTPUT_DIR}/datasets/${MODEL_NAME}

for dataset in "${datasets[@]}"; do
    torchrun --nproc_per_node=$NUM_GPUS generate.py \
        --base_model ${MODEL_PATH} \
        --dataset_name $dataset \
        --out_path  output_path\
        --batch_size $batch_size \
        --max_sample $max_sample \
        --threshold_path ../../threshold/${MODEL_NAME}/sparse-${SPARSE}.json
done

json_paths=()
for dataset in "${datasets[@]}"; do
    json_paths+=("${output_path}/${dataset}_T0.2_N1024_S42_${max_sample}.json")
done

# 合并 JSON 文件
OUTPUT_JSON="${output_path}/mix_wikitext_alpaca_c4_15000.json"
python mix_data.py --output "$OUTPUT_JSON" --inputs "${json_paths[@]}"

cd ../..


