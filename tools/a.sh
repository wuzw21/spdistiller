echo "SKU: $SKU"
if [ -z "$SKU" ]; then
    NUM_GPUS=4
    MAX_MEMORY="24000MB"
else
    NUM_GPUS=$(echo "$SKU" | sed -E 's/.*G([0-9]+)-.*/\1/')
    MAX_MEMORY_GB=$(echo "$SKU" | sed -E 's/([0-9]+)G.*/\1/')
    MAX_MEMORY=$((MAX_MEMORY_GB * 1000))MB
fi

echo "NUM_GPUS: $NUM_GPUS"
echo "MAX_MEMORY: $MAX_MEMORY"