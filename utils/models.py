from transformers import AutoModelForCausalLM
import time
from accelerate import infer_auto_device_map, dispatch_model
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
        max_memory = {0: "24GiB", 1: "24GiB", 2: "24GiB", 3: "24GiB", "cpu": "128GiB"}
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
    # model = dispatch_model(model, device_map=device_map)
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