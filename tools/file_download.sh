#!/bin/bash

# 定义基础路径
export BASE_URL="projects/wzw-gcrbitdistillerq4/amlt-results/7260484029.43713-028a1246-8bda-44a9-b674-6fd3b2b30944/ckpts/Meta-Llama-3-8B/int4-g64"

amlt storage list $BASE_URL
TARGET_DIR="/data/wzw/model/Meta-Llama-3-0.7-static"

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 定义文件列表
FILES=(
    "config.json"
    "generation_config.json"
    "model-00001-of-00004.safetensors"
    "model-00002-of-00004.safetensors"
    "model-00003-of-00004.safetensors"
    "model-00004-of-00004.safetensors"
    "model.safetensors.index.json"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "trainer_state.json"
    "training_args.bin"
)

# 下载每个文件
for FILE in "${FILES[@]}"; do
    echo "Downloading $BASE_URL/$FILE..."
    amlt storage download "$BASE_URL/$FILE" "$TARGET_DIR/$FILE"
done

echo "All files have been downloaded to $TARGET_DIR"