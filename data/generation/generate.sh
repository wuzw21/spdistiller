export MODEL_DIR=$1
export DATASET=$2
export OUTPUT=$3
export BATCH_SIZE=$4
export MAX_SAMPLE=$5
export NUM_GPUS=$6  # 第六个参数指定GPU数量

# 动态设置可见的 GPU 设备编号
GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# if [ $NUM_GPUS -gt 2 ]; then
#     torchrun --nproc_per_node ${NUM_GPUS} --master_port 7830 generate.py \
#         --base_model ${MODEL_DIR} \
#         --dataset_name ${DATASET} \
#         --out_path ${OUTPUT} \
#         --batch_size ${BATCH_SIZE} \
#         --max_sample ${MAX_SAMPLE}
# else
python single_generate.py \
    --base_model ${MODEL_DIR} \
    --dataset_name ${DATASET} \
    --out_path ${OUTPUT} \
    --batch_size ${BATCH_SIZE} \
    --max_sample ${MAX_SAMPLE}
# fi
