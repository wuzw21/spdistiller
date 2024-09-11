#!/bin/bash

export PATH=/home/aiscuser/.local/bin:${PATH}

pip uninstall -y onnxruntime_training

pip install -r requirement.txt

mkdir -p 3rdparty

cd 3rdparty

installBitsandbytes() {
    git clone https://github.com/FuchengJia1996/bitsandbytes-Q4_0.git "bitsandbytes"
    cd bitsandbytes
    bash tools/build.sh
    bash tools/install.sh
    cd ..
}

installTransformers() {
    git clone https://github.com/FuchengJia1996/transformers-pred.git "transformers"
    cd transformers
    pip install -e .
    cd ..
}

prepare() {
    wandb offline
}

installTransformers

installBitsandbytes

prepare
