#!/bin/bash

export DATASET_DIR=/data/wzw/Projects/BitDistiller/data/datasets

export MODEL_NAME=$1

cd data/generation

python mix_data.py

cd ../..
