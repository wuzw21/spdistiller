import json
import os
threshold_path = os.path.join("data", "threshold", "Meta-Llama-3-8B", "origin-threshold")
dict = [[{} for x in range(7)] for _ in range(32)]
for filename in os.listdir(threshold_path):
    # 检查文件名是否符合sparse-{x}.json的格式
    if filename.startswith("sparse-") and filename.endswith(".json"):
        # 提取x的值，假设x是文件名中sparse-和.json之间的部分
        x_str = filename.split("-")[1].split(".json")[0]
        print(x_str)
        with open(os.path.join(threshold_path, filename), 'r') as f:
            sparsity_all_dict = json.load(f)
            for i in range(32):
                layer_key = f"{i}"
                if layer_key in sparsity_all_dict:
                    layer_thresholds = sparsity_all_dict[layer_key]
                    for j in range(7):
                        dict[i][j][x_str] = layer_thresholds.get(f"{j}", 0.0)

for i in range(32):
    for j in range(7):
        print(f"Layer {i}, Weight {j}:", dict[i][j])
        pass

big_dict = {}
for i in range(32):
    big_dict[f"{i}"] = {}
    for j in range(7):
        big_dict[f"{i}"][f"{j}"] = dict[i][j]
with open(os.path.join(threshold_path, 'sparse_all.json'), 'w') as f:
    json.dump(big_dict, f)