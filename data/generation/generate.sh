
export MODEL_DIR=$1
export DATASET=$2
export OUTPUT=$3
export BATCH_SIZE=$4
export MAX_SAMPLE=$5

export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0

# torchrun --nproc_per_node ${NUM_GPUS} --master_port 7830 generate.py \
#                         --base_model ${MODEL_DIR} \
#                         --dataset_name ${DATASET} \
#                         --out_path ${OUTPUT} \
#                         --batch_size ${BATCH_SIZE} \
#                         --max_sample ${MAX_SAMPLE}

# Single Generate
python single_generate.py \
    --base_model ${MODEL_DIR} \
    --dataset_name ${DATASET} \
    --out_path ${OUTPUT} \
    --batch_size ${BATCH_SIZE} \
    --max_sample ${MAX_SAMPLE}
