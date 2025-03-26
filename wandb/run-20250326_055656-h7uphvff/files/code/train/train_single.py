import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(project_root)
from utils.sparse_hook import prepare_sparse_hook
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
from datasets import load_dataset
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import logging
from transformers import HfArgumentParser
# 设置日志记录
logging.basicConfig(level=logging.INFO)

# 定义模型参数类
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/data/wzw/models/Qwen2.5-0.5B-Instruct")

# 定义训练参数类
@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default="./results")
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    num_train_epochs: int = field(default=3)
    learning_rate: float = field(default=2e-5)
    logging_dir: str = field(default="./logs")
    logging_steps: int = field(default=10)
    evaluation_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    fp16: bool = field(default=True)
    use_lora: bool = field(default=True)
def preprocess_wiki(examples, tokenizer):
    # 获取批次中的文本列表
    texts = examples["text"]
    # 对文本进行 tokenization，填充到固定长度，并截断过长文本
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    # 返回字典，其中 labels 直接复制 input_ids
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].clone()
    }

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")
        self.data = self.data.map(lambda x: preprocess_wiki(x, tokenizer), batched=True)
        self.input_ids = torch.tensor(self.data["input_ids"])
        self.attention_mask = torch.tensor(self.data["attention_mask"])
        self.labels = torch.tensor(self.data["labels"])


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

# 定义训练函数
def train():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # 设置随机种子
    random.seed(training_args.seed)

    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    prepare_sparse_hook(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    # 准备模型进行LoRA训练
    if training_args.use_lora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 加载数据集
    dataset = CustomDataset(data_path="/home/donglinbai/Projects/wzw/BitDistiller-Q4_0/data/datasets/Qwen2.5-0.5B-Instruct/mix_wikitext_alpaca_c4_12.json", tokenizer=tokenizer)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)

    # 定义训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda data: {"input_ids": torch.stack([d["input_ids"] for d in data]), 
                                    "attention_mask": torch.stack([d["attention_mask"] for d in data]), 
                                    "labels": torch.stack([d["labels"] for d in data])},
    )

    # 开始训练
    trainer.train()

    # 保存模型
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()