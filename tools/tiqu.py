import re

# 输入文件路径
input_file = "results.txt"

# 定义需要提取的任务名称及其对应的列名
tasks_to_extract = {
    "gsm8k": "gsm8k",
    "arc_c": "arc_challenge",
    "arc_easy": "arc_easy",
    "agieval": "agieval",
    "piqa": "piqa"
}

# 初始化字典来存储提取的结果
results = {task: None for task in tasks_to_extract.keys()}

# 读取输入文件并提取信息
with open(input_file, "r") as file:
    lines = file.readlines()

# 遍历每一行，查找目标任务并提取值
for line in lines:
    for task_name, task_keyword in tasks_to_extract.items():
        if re.search(rf"\b{task_keyword}\b", line):
            match = re.search(r"\|\s*Value\s*\|\s*([\d\.]+)", line)
            if match:
                results[task_name] = match.group(1)

# 输出提取的结果
print("mmlu\tgsm8k\tarc_c\tarc_easy\tagieval\tpiqa")
print("\t".join(results.values()))