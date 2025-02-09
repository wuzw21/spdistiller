
import argparse
import numpy as np
import fnmatch
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Function
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--task', type=str, help='Task.')
    parser.add_argument('--limit', type=int, default=-1, help='Number of test samples.')
    parser.add_argument('--num_shot',type=int, default=0,help='NUM_SHOT')
    parser.add_argument('--sparse', type=float, default=0,help='sparse')
    parser.add_argument('--do_cr',type=int, default=0,help='cross_layer')
    parser.add_argument('--file_path',type=str, default=0,help='file_path')
    args = parser.parse_args()
    return args


def get_llm(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="balanced",
        trust_remote_code=True
    )

    if "70b" in model_path or "70B" in model_path:
        #max_memory = {0: "68GiB", 1: "80GiB", "cpu": "128GiB"}
        max_memory = {0: "36GiB", 1: "36GiB", 2: "36GiB", 3: "36GiB", "cpu": "128GiB"}
    elif "8x7B" in model_path:
        #max_memory = {0: "24GiB", 1: "80GiB", "cpu": "128GiB"}
        max_memory = {0: "32GiB", 1: "32GiB", 2: "32GiB", "cpu": "128GiB"}
    elif "13b" in model_path:
        max_memory = {0: "16GiB", 1: "16GiB", "cpu": "128GiB"}
        #max_memory = {0: "8GiB", 1: "8GiB", 2: "8GiB", 3: "8GiB", "cpu": "128GiB"}
    elif "7b" in model_path or "8B" in model_path:
        #max_memory = {0: "32GiB", "cpu": "128GiB"}
        #max_memory = {0: "8GiB", 1: "8GiB", "cpu": "128GiB"}
        #max_memory = {0: "1MiB", 1: "32GiB", "cpu": "128GiB"}
        max_memory = {0: "4GiB", 1: "4GiB", 2: "4GiB", 3: "4GiB", "cpu": "128GiB"}
    elif 'mini' in model_path :
        max_memory = {0: "4GiB", 1: "4GiB", 2: "4GiB", 3: "4GiB", "cpu": "128GiB"}
    else:
        max_memory = {0: "32GiB", 1: "32GiB", 2: "32GiB", "cpu": "128GiB"}
    print("max_memory:", max_memory)
    # kwargs = {"max_memory": max_memory} if len(max_memory) else {}
    kwargs = {}
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=[
            "OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "MixtralDecoderLayer", "DecoderLayer"],
        **kwargs
    )
    stime = time.time()
    #model = dispatch_model(model, device_map=device_map)
    print(f"load_time: {time.time() - stime:.3f} sec")

    model.seqlen = model.config.max_position_embeddings 

    model.config.pretraining_tp = 1
    model.config.use_cache = False
    model.config.output_hidden_states = False
    model.config.output_attentions = False

    model.config.do_sample = False
    model.config.temperature = 1.0
    model.config.top_p = 1.0
    model.config._attn_implementation_internal = "eager"
    print("model.hf_device_map:", model.hf_device_map)
    return model


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def quant_and_dequant_tensor_q4_0(inp, do_transpose=False):
    from bitsandbytes.functional import quantize_4bit, dequantize_4bit

    t = inp.permute(0, 1) if do_transpose else inp
    blocksize = 64
    quant_type = "fp4"
    q, quant_state = quantize_4bit(t, blocksize=blocksize, quant_type=quant_type)
    t = dequantize_4bit(q, quant_state, blocksize=blocksize, quant_type=quant_type)
    t = t.permute(0, 1) if do_transpose else t
    if t.size() != inp.size():
        raise ValueError(
            f"Tensor shape is not euqal. (t {t.size()} and inp {inp.size()})",
        )
    inp.data = t.data


class Floor(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


def quant_and_dequant_tensor_q4_0_v2(x, do_transpose=False):
    x = x.permute(0, 1) if do_transpose else x

    org_w_shape = x.shape

    q_group_size = 64
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        x = x.reshape(-1, q_group_size)
    assert x.dim() == 2

    # Quant.
    x = x.to(dtype=torch.float32, device=x.device)

    abs_max_val = (x.abs().amax(dim=1, keepdim=True)) / -8
    #print("abs_max_val:", abs_max_val)
    x = Floor.apply(torch.minimum(torch.ones_like(x, device=x.device) * 15.0, x * (1.0 / abs_max_val) + 8.5))
    #print("x:", x)

    # Dequant.
    x = (x - 8) * abs_max_val

    x = x.to(dtype=torch.bfloat16, device=x.device)

    x = x.reshape(org_w_shape)

    x = x.permute(0, 1) if do_transpose else x

    return x

def quant_and_dequant_model_q4_0(model):
    layers = model.model.layers
    for i in tqdm(range(0, len(layers)), desc="Quantizing"):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            #print(f"layer {i}, subset {name}")
            weight = None
            from_weights_map = False
            hf_hook = getattr(subset[name], "_hf_hook", None)
            if hf_hook is not None:
                weights_map = getattr(hf_hook, "weights_map", None)
                if weights_map is not None:
                    #print("Move weight to cuda")
                    weight = hf_hook.weights_map["weight"].to("cuda")
                    from_weights_map = True
            if weight is None:
                weight = subset[name].weight

            if name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] or \
                name in ["mlp.gate_proj", "mlp.up_proj"]:
                do_transpose = False
            elif name == "self_attn.o_proj" or name == "mlp.down_proj":
                do_transpose = True
                #do_transpose = False
            else:
                do_transpose = False

            #quant_and_dequant_tensor_q4_0(weight, do_transpose)
            quant_and_dequant_tensor_q4_0_v2(weight, do_transpose)

            if from_weights_map:
                weights_map["weight"] = weight.to("cpu")


def download_dataset(task_name):
    import datasets
    datasets.load_dataset(
        path=task_name,
        split="validation",
        name=None,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
    )


def eval(
    model_path,
    model,
    tokenizer,
    task_list=["boolq", "rte", "hellaswag", "winogrande", "arc_challenge", "arc_easy", "openbookqa"],
    num_fewshot=0,
    limit=None
):
    #from .llm_eval_adaptor import LMEvalAdaptor
    from lm_eval import tasks, evaluator
    from lm_eval.utils import make_table
    from lm_eval.tasks import TaskManager
    from lm_eval.models.huggingface import HFLM
    def pattern_match(patterns, source_list):
        # print(patterns,source_list)
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    #task_names = pattern_match(task_list, tasks.ALL_TASKS)
    task_manager = TaskManager()
    task_names = pattern_match(task_list, task_manager.all_tasks)
    print('tasks : ', task_names)
    batch_size = 1
    #lm_eval_model = LMEvalAdaptor(model_path, model, tokenizer, batch_size, max_length=-1)
    lm_eval_model = HFLM(model, tokenizer=tokenizer, batch_size=batch_size, max_length=4096)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        #model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=None,
        # no_cache=True,
        use_cache=None,
        limit=limit,
        #description_dict={},
        #decontamination_ngrams_path=None,
        check_integrity=False,
        #pretrained_model=model,
        #tokenizer=tokenizer,
        #add_special_tokens=add_special_tokens
    )
    #print(results["results"]["wikitext"]["perplexity"])

    return make_table(results)


def eval_for_sp_config(model_path, model, tokenizer, task_list, num_shot, limit, sp_config, file_path):
    print("sp_config: ", sp_config, "num_shot: ", num_shot)
    global_weight_preditor = model.model.global_weight_preditor
    global_weight_preditor.reset()
    # global_weight_preditor = None
    if global_weight_preditor is not None:
        attn_sp, mlp_sp, w_p, do_cr = sp_config
        global_weight_preditor.set_sp_config(attn_sp, mlp_sp, w_p)
        global_weight_preditor.set_do_pre_prediction(do_cr)
    global_weight_preditor.set_sparsity_threshold(file_path)
    results = eval(
        model_path, model, tokenizer, task_list,
        num_shot, limit
    )
    print('='*40)
    print("sp_config: ", sp_config, "num_shot: ", num_shot)
    print("results")
    print(results)
    print('='*40)


def main():
    args = parse_args()

    #download_dataset(args.task)
    #return

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"loading llm model {args.model}")
    model = get_llm(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    quant_and_dequant_model_q4_0(model)

    print("task: eval")

    #task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
    #task_list = ["boolq"]
    #task_name = os.environ["EVAL_DATASET"]
    task_name = args.task
    print(f"task_name: {task_name}")
    #limit = None
    #limit = 100
    limit = args.limit
    if limit is None or limit <= 0:
        limit = None
    task_list = [args.task, 'wikitext']
    num_shots= list(set([args.num_shot,0]))
    if "70b" in args.model or "70B" in args.model:
        sp_configs = [(0.00, 0.00, 0.00, 0), (0.50, 0.50, 0.00, 0), (0.90, 0.90, 0.00, 0)]
    else:
        print(args.sparse,args.do_cr)
        sp_configs = list(set([(args.sparse, args.sparse, 0.00, args.do_cr), (0.50, 0.50, 0.00, 0), (0.70, 0.70, 0.00, 0)]))
    print('sp_configs: ',sp_configs)
    for sp_config in sp_configs:
        for num_shot in num_shots: 
            eval_for_sp_config(args.model, model, tokenizer, task_list, num_shot, limit, sp_config, args.file_path)


if __name__ == "__main__":
    main()