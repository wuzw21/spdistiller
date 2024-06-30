

export MODEL_NAME=Llama-2-7b-chat-hf

cd train

bash train.sh \
    ../data/datasets/${MODEL_NAME}/mix_wiki_alpaca_80.json \
    ./ckpts/${MODEL_NAME}/int4-g64/ \
    ./logs/${MODEL_NAME}/int4-g64/ 4

cd ..
