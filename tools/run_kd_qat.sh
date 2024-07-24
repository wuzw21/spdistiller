

export MODEL_NAME=Meta-Llama-3-8B

cd train

bash train.sh \
    ../data/datasets/${MODEL_NAME}/c4_T0.7_N1024_S42_4096.json \
    ./ckpts/${MODEL_NAME}/int4-g64/ \
    ./logs/${MODEL_NAME}/int4-g64/ 4

cd ..
