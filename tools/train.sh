#!/bin/bash

cd train

bash train.sh \
    ${DATA_DIR}/datasets/${MODEL_NAME}/${DATASET} \
    ${OUTPUT_DIR}/ckpts/${MODEL_NAME}/ \
    ${OUTPUT_DIR}/logs/${MODEL_NAME}/
cd ..