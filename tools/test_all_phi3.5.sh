# 模型名称数组
models=(
  "phi-3.5-sparse_0.5-cr_1_wac8k"
  "phi-3.5-sparse_0.8-cr_1_wac8k"
  "phi-3.5-sparse_0.9-cr_1_wac8k"
  "phi-3.5-sparse_0.5-cr_0_wac8k"
  "phi-3.5-sparse_0.8-cr_0_wac8k"
  "phi-3.5-sparse_0.9-cr_0_wac8k"
)

sparses=(
    0.5
    0.8
    0.9
    0.5
    0.8
    0.9
)

crs=(
    1
    1
    1
    0
    0
    0
)

for i in "${!models[@]}"
do
  model="${models[i]}"
  sparse="${sparses[i]}"
  cr="${crs[i]}"
  echo $model
  echo $sparse
  echo $cr
  amlt map --sla Premium bitdistiller.yaml :gcrbitdistiller_test_task "$model" --description "mmlu-$model" --extra-args "$sparse $cr"
done