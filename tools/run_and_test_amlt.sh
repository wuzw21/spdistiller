SKU=${SKU:-"40G4-A100"}

export NUM_GPUS=$(echo "$SKU" | sed -E 's/.*G([0-9]+)-.*/\1/')

echo "NUM_GPUS: $NUM_GPUS"

export MAX_MEMORY=$(echo "$SKU" | sed -E 's/^([0-9]+)G.*/\1/' | awk '{print $1 * 10000}')

echo "MAX_MEMORY: $MAX_MEMORY MB"


export MODEL_NAME=Meta-Llama-3.1-8B-Instruct
export MODEL_PATH=${AMLT_DATA_DIR}/models/${MODEL_NAME}
export MODEL_SAVE_PATH=${AMLT_OUTPUT_DIR}/ckpts/${MODEL_NAME}/int4-g64/
#export TEMPERATURE=0.7
export TEMPERATURE=0.2

export NUM_TRAIN_EPOCHS=4

export NUM_NODES=1

export ATTN_SP=0.5
export MLP_SP=0.5
export W_P=0.0
export DO_CR=0

cd train

bash train_amlt.sh \
    ${AMLT_DATA_DIR}/datasets/${MODEL_NAME}/mix_wikitext_alpaca_c4_14000.json \
    ${AMLT_OUTPUT_DIR}/ckpts/${MODEL_NAME}/int4-g64/ \
    ${AMLT_OUTPUT_DIR}/logs/${MODEL_NAME}/int4-g64/ \
    ${NUM_TRAIN_EPOCHS} \
    ${MODEL_PATH} \
    ${MODEL_NAME} \
    ${NUM_NODES} \
    ${NUM_GPUS} \
    ${MAX_MEMORY}

cd ..


export TEST_TASK=wikitext
export ENABLE_PREDICTOR=1
export ENABLE_SPARSE_INFER=1
export ENABLE_TENSOR_SAVER=0
# export PREDICTOR_DATA_HOME=${HF_HOME}/predictor-data
# export PREDICTOR_DATA_DIR=${PREDICTOR_DATA_HOME}/${MODEL_NAME}-c4-sparse
# export PREDICT_CKPT_HOME=/data/fuchengjia/Projects/llm-wanda/checkpoints/weight-predictors

export LOCAL_RANK=-1

# unused parameters
export HF_HOME=/home/donglinbai/Projects/wzw
export PREDICTOR_DATA_HOME=${HF_HOME}/predictor-data
export PREDICTOR_DATA_DIR=${PREDICTOR_DATA_HOME}/${MODEL_NAME}-c4-sparse
export PREDICT_CKPT_HOME=/data/fuchengjia/Projects/llm-wanda/checkpoints/weight-predictors
export PROSPARSE_PREDICTOR=0
export PROSPARSE_PREDICTOR_DIR=${HF_HOME}/models/${MODEL_NAME}-predictor
export HF_DOWNLOAD_DATASET_HOME=${HF_HOME}/datasets
export HF_ENDPOINT=https://huggingface.co

echo "Model: ${AMLT_OUTPUT_DIR}/ckpts/${MODEL_NAME}/int4-g64/"

echo "CUDA device: ${CUDA_VISIBLE_DEVICES}"

cd test

#--limit=500
python test_task.py \
    --model=${MODEL_SAVE_PATH} \
    --seed=42 \
    --task=${TEST_TASK}
cd ..
