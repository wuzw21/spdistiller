import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM,AutoTokenizer
import re

def sparse_hook(module, input, predictor, layer_id, weight_id):
    # print(input)
    if predictor == None:
        return input
    else :
        sp_x = predictor.generate_pred(layer_id, weight_id, input[0])
        return sp_x  

def get_layer_id_from_prefix(prefix) -> int:
    match = re.search(r'\blayers\.(\d+)\b', prefix)
    return int(match.group(1)) if match else None

def traverse_and_register_hooks(model):
    if not hasattr(model, 'predictor'):
        model.predictor = None
        
    weight_map = []   # (layer_id, weight_name, weight_id)
    weight_counters = {}  # count weight_id
    num_layers = 0
    def _traverse(module, prefix="", layer_id=None):
        nonlocal num_layers
        # get layer id
        extracted_layer_id = get_layer_id_from_prefix(prefix)
        if extracted_layer_id is not None:
            new_layer_id = extracted_layer_id
            num_layers = max(num_layers , extracted_layer_id + 1)
        else:
            new_layer_id = layer_id if layer_id is not None else "global"

        # init weight counter
        if new_layer_id not in weight_counters:
            weight_counters[new_layer_id] = 0

        # register hook
        if isinstance(module, nn.Linear):
            current_weight_id = weight_counters[new_layer_id]
            weight_counters[new_layer_id] += 1
            weight_map.append((new_layer_id, prefix, current_weight_id))
            # register
            if new_layer_id != 'global' :
                module.register_forward_pre_hook(
                    lambda module, input, layer_id=new_layer_id, weight_id=current_weight_id:
                        sparse_hook(module, input, model.predictor, layer_id, weight_id)
                )

        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            _traverse(child, child_prefix, new_layer_id)

    _traverse(model)

    if model.predictor != None :
        weight_counters.pop('global')
        model.predictor.set_layers_and_weights(num_layers, weight_counters, weight_map)
    return weight_map



def prepare_sparse_hook(model) :
    print(model)
    print("Printing model layers...")
    weight_map = traverse_and_register_hooks(model)
    #   print(weight_map)

def main() :
    # 创建模型
    # model_path = "/data/wzw/models/Mixtral-8x7B-Instruct-v0.1"
    model_path = "/data/wzw/models/Meta-Llama-3-8B" 
    model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prepare_sparse_hook(model)

    while(True) :
        i = input('>')
        inputs = tokenizer(i, return_tensors="pt").to(model.device)

        outputs = model.generate(inputs['input_ids'], max_length=100, do_sample=True)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)

    
if __name__ == '__main__' :
    main()