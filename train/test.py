import datasets
datasets.load_dataset(
    "tasksource/mmlu",
    "abstract_algebra",
    split="validation",
)
print(datasets)