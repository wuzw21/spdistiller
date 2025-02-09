# 模型名称数组
models=(
  "gcrbitdistillerllama-3.1-8b-chat-40G4-A100-0.7-do_cr"
  "gcrbitdistillerllama-3.1-8b-chat-40G4-A100-0.6-do_cr"
  # "llama-3.1-sparse_0.5-cr_1_wac14k"
  # "gcrbitdistillerllama-3.1-8b-chat-40G4-A100-sparse_0.9-do_cr"
  # "gcrbitdistillerllama-3.1-8b-chat-40G4-A100-sparse_0.8-do_cr"
  # "gcrbitdistillerllama-3.1-8b-chat-40G4-A100-sparse_0.7-do_cr"
  # "gcrbitdistillerllama-3.1-8b-chat-40G4-A100-sparse_0.6-do_cr"
  # "gcrbitdistillerllama-3.1-8b-chat-40G4-A100-sparse_0.5-do_cr"
  # "gcrbitdistillerllama-3.1-8b-chat-40G4-A100-0.9"
  # "gcrbitdistillerllama-3.1-8b-chat-40G4-A100-0.8"
  "gcrbitdistillerllama-3.1-8b-chat-40G4-A100-0.7"
  "gcrbitdistillerllama-3.1-8b-chat-40G4-A100-0.6"
  # "gcrbitdistillerllama-3.1-8b-chat-40G4-A100"
)

sparses=(
  0.7
  0.6
  0.7
  0.6
    # 0.5
    # 0.9
    # 0.8
    # 0.7
    # 0.6
    # 0.5
    # 0.9
    # 0.8
    # 0.5
)

crs=(
    1
    1
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