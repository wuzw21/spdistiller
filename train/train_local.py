import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(project_root)
from quantization.qlinear import QLinear, convertModelToQuant
from quantization.clip_utils import apply_clip

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, BitsAndBytesConfig, default_data_collator
from datasets import load_dataset
import json
import glob
import torch.distributed as dist

from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from mytrainer import KDTrainer
import random
from tqdm import tqdm
from datasets import load_dataset

from utils.sparse_hook import prepare_sparse_hook, get_sparsity_configs

# 如果需要使用LoRA，则导入PEFT相关模块
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training

def get_last_checkpoint(output_dir: str) -> Optional[str]:
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda ckpt: int(ckpt.split("-")[-1]))
    return checkpoints[-1]

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    threshold_path: str = field(default=None, metadata={"help": "Path to save threshold"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=False)
    bits: int = field(default=2, metadata={"help": "How many bits to use."})
    q_group_size: int = field(default=128, metadata={"help": "Quantization Group Size."})
    quant_type: str = field(
        default="int2-asym",
        metadata={"help": "Quantization data type to use. Should be one of `int2-asym` or `ste-n2f3`."}
    )
    clip: str = field(default=None, metadata={"help": "The path of clip cache"})
    train_kd: bool = field(default=False, metadata={"help": "Whether to use KD to QAT"})
    kd_tmp: int = field(default=1, metadata={"help": "Temperature of KD"})
    kd_loss_type: str = field(default=None, metadata={"help": "Type of loss function when KD-QAT"})
    cakld_steps: int = field(default=10, metadata={"help": "How many steps to calculate the coefficient of CAKLD."})
    max_memory: str = field(default="80000MB", metadata={"help": "max_memory for cuda device"})
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy to adopt during training. Options: 'no', 'steps', 'epoch'."}
    )
    eval_steps: int = field(default=500, metadata={"help": "Number of update steps between two evaluations."})
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA for fine-tuning."})

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_sample: int, split: str):
        super().__init__()
        with open(data_path, 'r') as f:
            lines = f.readlines()
        all_dataset = [json.loads(line.strip()) for line in lines]
        sources, targets = zip(*[(s[0][0], f"{s[0][1]}{tokenizer.eos_token}") for s in all_dataset])
        dataset_size = len(sources)
        max_sample = min(max_sample or dataset_size, dataset_size)
        if max_sample < dataset_size:
            indices = random.sample(range(dataset_size), max_sample)
            self.sources, self.targets = [sources[i] for i in indices], [targets[i] for i in indices]
        else:
            self.sources, self.targets = sources, targets
        split_num = len(self.sources) // 5
        if split == "train":
            self.sources, self.targets = self.sources[split_num:], self.targets[split_num:]
            print(f"Using {len(self.sources)} samples to train")
            print("Example Data")
            print("sources: \n", self.sources[0])
            print("targets: \n", self.targets[0])
        elif split == "eval":
            self.sources, self.targets = self.sources[:split_num], self.targets[:split_num]
            print(f"Using {len(self.sources)} samples to evaluation")
    def __len__(self):
        return len(self.sources)
    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            sources.append(instance['input_ids'])
            targets.append(instance['labels'])
        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, max_sample=data_args.max_train_samples, split="train")
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, max_sample=data_args.max_train_samples, split="eval")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def get_int_from_envs(name):
    var = os.environ.get(name)
    return int(var) if var is not None else 0

def get_float_from_envs(name):
    var = os.environ.get(name)
    return float(var) if var is not None else 0

import math
def compute_metrics(eval_pred):

    try:
        loss = eval_pred.metrics["eval_loss"]
        if loss is not None:
            ppl = math.exp(loss)
        else:
            ppl = float("inf")
    except Exception as e:
        ppl = float("inf")

    return {"ppl": ppl}

def train():
    # try:
    #     print(f"ROCm是否可用: {torch.cuda.is_available()}")
    #     print(f"设备数量: {torch.cuda.device_count()}")
    #     print(f"当前设备: {torch.cuda.current_device()}")
    #     print(f"设备名称: {torch.cuda.get_device_name(0)}")
    #     print(torch.__version__)
    #     print(torch.cuda.is_available())
    #     print(torch.version.hip)
    # except KeyError as e:
    #     pass

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    random.seed(training_args.seed)
    n_gpus = torch.cuda.device_count()
    max_memory = training_args.max_memory
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}
    print('max_memory:', max_memory, "device_map", device_map)
    print(f"loading {model_args.model_name_or_path} model")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model = prepare_model_for_kbit_training(model)
    # prepare sparse
    prepare_sparse_hook(model)
    global_weight_preditor = model.predictor
    if global_weight_preditor is not None:
        attn_sp, mlp_sp, w_p, do_cr = get_sparsity_configs()
        global_weight_preditor.set_sp_config(attn_sp, mlp_sp, w_p)
        global_weight_preditor.set_do_pre_prediction(do_cr)
        global_weight_preditor.set_sparsity_threshold(data_args.threshold_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    pad_status = True
    if tokenizer.pad_token is None:
        print("tokenizer has not padding token")
        pad_status = False
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # load quantization if specified
    if training_args.quant_type is not None:
        print("converting the model to qat, this may take a while...")
        # model, _ = convertModelToQuant(model, compute_dtype=torch.bfloat16, quant_type=training_args.quant_type, q_group_size=training_args.q_group_size)
    if training_args.clip is not None:
        # q_config = {"zero_point": True, "q_group_size": training_args.q_group_size}
        # print("Loading pre-computed Clipping results from", training_args.clip)
        # clip_results = torch.load(training_args.clip)
        # apply_clip(model, clip_results)
        print("Clipping init successfully!")
        
    # 如果需要进行LoRA微调，则包装模型
    if training_args.use_lora:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.1,
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            target_modules=["q_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        print("Applying LoRA fine-tuning...")
    
    # load teacher model if KD training is enabled (assume LoRA and KD are mutually exclusive)
    if training_args.train_kd:
        print("loading Teacher Model...")
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_4bit=False,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            max_memory=max_memory,
        )
        teacher_model.eval()
        teacher_model.cuda()
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.config.use_cache = False
        if pad_status is False:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=teacher_model,
            )
        model.kd_loss_scale = 1.0
        print("Teacher Model loaded")

    # compute cakld coefficient if needed
    mean_prob = 0
    if training_args.train_kd and training_args.kd_loss_type == "cakld":
        print("Get the main Prob!")
        probDataloader = DataLoader(
            data_module['train_dataset'], 
            shuffle=True, 
            collate_fn=data_module['data_collator'], 
            batch_size=training_args.per_device_train_batch_size,
            drop_last=True,
        )
        prob = 0
        for step, batch in tqdm(enumerate(probDataloader)):
            if step > training_args.cakld_steps:
                break
            batch = {k: v.to(teacher_model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = teacher_model(**batch)
            logits = outputs.get("logits").contiguous()
            prob1 = torch.nn.functional.softmax(logits, dim=-1)
            prob1 = torch.max(prob1, dim=-1).values 
            prob += prob1.mean()
        mean_prob = prob / training_args.cakld_steps
        mean_prob = torch.Tensor(mean_prob.to(teacher_model.device))
        dist.all_reduce(mean_prob, op=dist.ReduceOp.SUM)
        mean_prob = mean_prob / dist.get_world_size()
        print(f"Get the coefficient: {mean_prob}")

    # load trainer (这里假设LoRA和KD微调互斥)
    if training_args.train_kd:
        trainer = KDTrainer(model=model, tokenizer=tokenizer, teacher_model=teacher_model, loss_type=training_args.kd_loss_type, mean_prob=mean_prob, args=training_args, compute_metrics=compute_metrics, **data_module)
    else:
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, compute_metrics=compute_metrics, **data_module)

    resume_ckpt = get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
