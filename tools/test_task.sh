#!/bin/bash

cd train

#--limit=500
python test_task.py \
    --model=${MODEL_PATH} \
    --seed=42 \
    --task=${TEST_TASK} \
    --sparse=${SPARSE} \
    --limit=10 \
    --num_shot=0 \
    --do_cr=${DO_CR} \
    --threshold_path=${THRESHOLD_PATH} \
    --sparse_strategy=${SPARSE_STRATEGY} \
    --batch_size=1 \
    --quant=1 \
    --test_all=${TEST_ALL}
cd ..