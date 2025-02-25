import os
import json
import copy
import random
import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from transformers import Trainer, TrainingArguments, default_data_collator, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data (JSON file)."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "For debugging, limit the number of training samples."})

@dataclass
class LoRATrainingArguments(TrainingArguments):
    model_max_length: int = field(
        default=2048, 
        metadata={"help": "Maximum sequence length. Sequences will be right-padded/truncated."}
    )
    cache_dir: Optional[str] = field(default=None)
    evaluation_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy: 'no', 'steps', 'epoch'."})
    eval_steps: int = field(default=500, metadata={"help": "Steps between evaluations."})
    output_dir: str = field(default="./lora_finetuned")
    overwrite_output_dir: bool = field(default=False)

def safe_save_model(trainer: Trainer, output_dir: str):
    """保存模型参数到磁盘"""
    state_dict = trainer.model.state_dict()
    cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
    trainer._save(output_dir, state_dict=cpu_state_dict)

def get_int_from_envs(name):
    var = os.environ.get(name)
    if var is None:
        return 0
    else:
        return int(var)

def get_float_from_envs(name):
    var = os.environ.get(name)
    if var is None:
        return 0
    else:
        return float(var)
def get_sparsity_configs():
    attn_sp = get_float_from_envs("ATTN_SP")
    mlp_sp = get_float_from_envs("MLP_SP")
    w_p = get_float_from_envs("W_P")
    do_cr = get_int_from_envs("DO_CR")
    return attn_sp, mlp_sp, w_p, do_cr
    
def preprocess_function(example, tokenizer):
    # 假设每个 JSON 对象包含 "source" 和 "target" 字段，将二者拼接
    text = example["source"] + " " + example["target"] + tokenizer.eos_token
    tokenized = tokenizer(text, truncation=True, max_length=tokenizer.model_max_length)
    return tokenized
    
def train():
# 使用 bitsandbytes 进行 4bit 量化加载模型
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, LoRATrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 加载模型和分词器
    print(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # 如果分词器没有 pad_token，则添加
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})
        model.resize_token_embeddings(len(tokenizer))
    
    global_weight_preditor = model.model.global_weight_preditor
    if global_weight_preditor is not None:
        attn_sp, mlp_sp, w_p, do_cr = get_sparsity_configs()
        global_weight_preditor.set_sp_config(attn_sp, mlp_sp, w_p)
        global_weight_preditor.set_do_pre_prediction(do_cr)
        global_weight_preditor.set_sparsity_threshold(data_args.threshold_path)
    
    # 配置 LoRA 参数（针对因果语言模型任务）
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",   # 保持自回归任务
        r=8,                     # LoRA 秩，可根据需要调整
        lora_alpha=32,           # Scaling 参数
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA modules added to the model.")
    
    # 加载数据集（假设数据为 JSON 文件，每行一个 JSON 对象，包含 "source" 和 "target" 字段）
    dataset = load_dataset("json", data_files={"train": data_args.data_path}, split="train")
    if data_args.max_train_samples is not None:
        dataset = dataset.select(range(data_args.max_train_samples))
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=False)
    
    # 使用默认的数据收集器
    data_collator = default_data_collator
    
    # 构建 Trainer 并开始微调
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    
    # 保存微调后的模型
    safe_save_model(trainer, training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__" :
    train()