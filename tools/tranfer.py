import os
import json

# 定义键名到数字的映射
key_mapping = {
    "q": 0,
    "k": 1,
    "v": 2,
    "o": 3,
    "gate": 4,
    "up": 5,
    "down": 6
}

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 创建一个新的字典，用于存储转换后的数据
        new_data = {}
        for outer_key, inner_dict in data.items():
            new_inner_dict = {}
            for key, value in inner_dict.items():
                if key in key_mapping:
                    new_key = key_mapping[key]
                    new_inner_dict[new_key] = value
            new_data[outer_key] = new_inner_dict

        # 将转换后的数据写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=4)
        print(f"Processed file: {file_path}")
    except Exception as e:
        print(f"Failed to process file: {file_path}. Error: {e}")

def traverse_and_process(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                process_file(file_path)

# 设置要处理的目录路径
directory_path = '/home/donglinbai/Projects/wzw/BitDistiller-Q4_0/threshold/Meta-Llama-3-8B'
traverse_and_process(directory_path)
