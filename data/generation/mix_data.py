
import os
import json
import random

all_outputs = []

model_name = os.environ["MODEL_NAME"]

json_path1 = None
#json_path1 = f"../datasets/{model_name}/wikitext_T0.7_N1024_S42_3000.json"
json_path2 = f"../datasets/{model_name}/alpaca_T0.2_N1024_S42_5000.json"
json_path3 = f"../datasets/{model_name}/c4_T0.2_N1024_S42_4096.json"

if json_path1 is not None:
    with open(json_path1, 'r') as f:
        dataset_for_eval = f.readlines()
    for line in dataset_for_eval:
        json_data = json.loads(line)
        all_outputs.append(json_data)

with open(json_path2, 'r') as f:
    dataset_for_eval = f.readlines()
for line in dataset_for_eval:
    json_data = json.loads(line)
    all_outputs.append(json_data)

with open(json_path3, 'r') as f:
    dataset_for_eval = f.readlines()
for line in dataset_for_eval:
    json_data = json.loads(line)
    all_outputs.append(json_data)

random.shuffle(all_outputs)

with open(f'../datasets/{model_name}/mix_alpaca_c4_9000.json', 'w') as f:
    for item in all_outputs:
        f.write(json.dumps(item) + '\n')
