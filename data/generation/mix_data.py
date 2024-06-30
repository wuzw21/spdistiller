
import os
import json
import random

all_outputs = []

model_name = os.environ["MODEL_NAME"]

json_path1 = f"../datasets/{model_name}/wikitext_T0.7_N1024_S42_30.json"
json_path2 = f"../datasets/{model_name}/alpaca_T0.7_N1024_S42_50.json"

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

random.shuffle(all_outputs)

with open(f'../datasets/{model_name}/mix_wiki_alpaca_80.json', 'w') as f:
    for item in all_outputs:
        f.write(json.dumps(item) + '\n')
