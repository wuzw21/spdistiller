#!/bin/bash

export PATH=/home/aiscuser/.local/bin:${PATH}

prepareSSHKey() {
    cp ssh/* ~/.ssh
}

prepareSSHKey

pip uninstall -y onnxruntime_training

pip install -r requirement.txt

mkdir -p 3rdparty

cd 3rdparty

installTransformers() {
    git clone -b wzw https://github.com/FuchengJia1996/transformers-pred.git "transformers"
    cd transformers
    pip install -e .
    cd ..
}

installLMEval() {
    git clone https://github.com/FuchengJia1996/lm-evaluation-harness-v0.4.3.git "lm-evaluation-harness"
    cd lm-evaluation-harness
    pip install -e .
    cd ..
}

prepare() {
    wandb offline
}

installTransformers

installLMEval

prepare
