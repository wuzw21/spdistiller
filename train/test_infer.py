
import sys
sys.path.append("../quantization")
import torch
import transformers
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict
from qlinear import convertModelToQuant


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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
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
    bits: int = field(
        default=2,
        metadata={"help": "How many bits to use."}
    )
    q_group_size: int = field(
        default=128,
        metadata={"help": "Quantization Group Size."}
    )
    quant_type: str = field(
        default="int2-asym",
        metadata={"help": "Quantization data type to use. Should be one of `int2-asym` or `ste-n2f3`."} # see quantization/qlinear.py
    )
    clip: str = field(
        default=None,
        metadata={"help": "The path of clip cache"}
    )
    train_kd: bool = field(default=False, metadata={"help": 'Whether to use KD to QAT'})
    kd_tmp: int = field(
        default=1,
        metadata={"help": "Temperature of KD"}
    )
    kd_loss_type: str = field(
        default=None,
        metadata={"help": "Type of loss function when KD-QAT"}
    )
    cakld_steps: int = field(
        default=10,
        metadata={"help": "How many step to caculate the coefficient of CAKLD."}
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def calc_ppl(loss):
    return torch.exp(loss)


def calc_ppl_promt(model, tokenizer, promt):
    input_ids = tokenizer.encode(promt, add_special_tokens=False, return_tensors="pt")
    device = next(iter(model.parameters())).device

    input_ids = input_ids.to(device)
    outputs = model(input_ids, labels=input_ids, return_dict=False)
    loss = outputs[0]
    ppl = calc_ppl(loss)
    return ppl, loss


def load_dataset_encodings(dataset_name, tokenizer):
    from datasets import load_dataset
    if dataset_name == "wikitext":
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    elif dataset_name == "c4":
        train = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train")
        train = train.select([*range(0, 100, 1)])
        encodings = tokenizer("\n\n".join(train["text"]), return_tensors="pt")
    return encodings


def calc_ppl_dataset(model, tokenizer, dataset_name):
    from tqdm import tqdm

    encodings = load_dataset_encodings(dataset_name, tokenizer)

    max_length = model.config.max_length
    stride = 512
    seq_len = encodings.input_ids.size(1)
    device = next(iter(model.parameters())).device

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids, return_dict=True)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    mean_loss = torch.stack(nlls).mean()
    ppl = torch.exp(mean_loss)
    return ppl, mean_loss


def cakld_loss(labels, student_logits, teacher_logits, beta_prob):
    from torch.nn import functional as F, MSELoss

    mask = (labels != -100)

    # reverse
    teacher_output_log_prob = F.log_softmax(teacher_logits, dim=2)
    # Compute the softmax of the student's logits (approximate distribution)
    student_output_soft = F.softmax(student_logits, dim=2)
    # Calculate the reverse KL Divergence (KL(teacher_logits || student_logits))
    reverse_kl = F.kl_div(teacher_output_log_prob, student_output_soft, reduction="none").sum(-1)

    # forward
    student_output_log_prob = F.log_softmax(student_logits, dim=2)
    teacher_output_soft = F.softmax(teacher_logits, dim=2)
    # Calculate the reverse KL Divergence (KL(teacher_logits || student_logits))
    forward_kl = F.kl_div(student_output_log_prob, teacher_output_soft, reduction="none").sum(-1)

    kl_loss = beta_prob * reverse_kl + (1 - beta_prob) * forward_kl
    kl_loss *= mask
    kl_loss = kl_loss.sum(-1).mean()
    return kl_loss


def calc_ppl_teacher_dataset(model, teacher_model, tokenizer, training_args, data_args):
    from tqdm import tqdm
    from utils import make_supervised_data_module
    from torch.utils.data import Dataset, DataLoader

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    data_loader = DataLoader(
        data_module['train_dataset'], 
        shuffle=True, 
        collate_fn=data_module['data_collator'], 
        batch_size=training_args.per_device_train_batch_size,
        drop_last=True,
    )

    prob = 0
    for step, batch in tqdm(enumerate(data_loader)):
        if step > training_args.cakld_steps:
            break
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = teacher_model(**batch)
        logits = outputs.get("logits").contiguous()
        prob1 = torch.nn.functional.softmax(logits, dim=-1)
        prob1 = torch.max(prob1, dim=-1).values 
        prob += prob1.mean()

    mean_prob = prob / training_args.cakld_steps
    mean_prob = torch.Tensor(mean_prob.to(teacher_model.device))

    nlls = []
    kl_nlls = []
    for step, batch in tqdm(enumerate(data_loader)):
        if step > 25:
            break
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            student_outputs = model(**batch)
            teacher_outputs = teacher_model(**batch)
        student_logits = student_outputs.get("logits").contiguous()
        teacher_logits = teacher_outputs.get("logits").contiguous()
        #print(f"student_logits_shape {student_logits.shape}, teacher_logits_shape {teacher_logits.shape}")
        neg_log_likelihood = student_outputs.get("loss")
        nlls.append(neg_log_likelihood)
        kl_loss = cakld_loss(batch["labels"], student_logits, teacher_logits, mean_prob)
        kl_nlls.append(kl_loss)
    mean_loss = torch.stack(nlls).mean()
    ppl = torch.exp(mean_loss)
    mean_kl_loss = torch.stack(kl_nlls).mean()
    return ppl, mean_loss, mean_kl_loss


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device_map = "auto"
    #device_map = "balanced"

    print(f"loading {model_args.model_name_or_path} model")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

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
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    #print("converting the model to qat, this may take a while...")
    #qmodel, _ = convertModelToQuant(model, compute_dtype=torch.bfloat16, quant_type=training_args.quant_type, q_group_size=training_args.q_group_size)
    qmodel = model
    # quant_and_dequant_model_q4_0(qmodel)

    print("loading Teacher Model...")
    teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_4bit=False,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    teacher_model.eval()
    #teacher_model.cuda()
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

    #promt = "Hello, I'm a language model"
    #ppl, loss = calc_ppl_promt(model, tokenizer, promt)
    #ppl, loss = calc_ppl_dataset(model, tokenizer, dataset_name="wikitext")
    #ppl, loss = calc_ppl_dataset(model, tokenizer, dataset_name="c4")
    ppl, loss, kl_loss = calc_ppl_teacher_dataset(qmodel, teacher_model, tokenizer, training_args, data_args)
    print(f"ppl {ppl}, loss {loss}, kl_loss {kl_loss}")


if __name__ == "__main__":
    main()
