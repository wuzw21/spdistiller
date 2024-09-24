#!/bin/bash

export AMLT_HOME=/data/fuchengjia2/Downloads/amlt

export BITDISTILLER_DATA_DATASETS_DIR=/data/fuchengjia2/Projects/BitDistiller/data/datasets

cd ${AMLT_HOME}

export MODEL_JOB_NAME=llama3170b
export MODEL_NAME=Meta-Llama-3.1-70B-Instruct

cp -r gcrbitdistillergenerateteacherdata${MODEL_JOB_NAME}/gcrbitdistillergenerateteacherdata/datasets/${MODEL_NAME} \
    ${BITDISTILLER_DATA_DATASETS_DIR}
