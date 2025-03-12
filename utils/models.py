    from transformers import AutoModelForCausalLM
    import time
    from accelerate import infer_auto_device_map, dispatch_model
    import os
    import torch
    from transformers import AutoModelForCausalLM, infer_auto_device_map

    def get_available_gpu_memory(device=0):
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(device)  # 返回值以字节为单位
            free_gib = free / (1024 ** 3)
            total_gib = total / (1024 ** 3)
            return free_gib, total_gib
        else:
            return None, None

    def get_max_memory():
        max_memory = {}
        num_devices = torch.cuda.device_count()
        for device in range(num_devices):
            free_gib, total_gib = get_available_gpu_memory(device)
            if free_gib is not None:
                allocation = free_gib * 0.9  # 分配90%空闲内存
                max_memory[device] = f"{allocation:.0f}GiB"
        max_memory["cpu"] = "128GiB"
        return max_memory

    def get_llm(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="balanced",
            trust_remote_code=True
        )

        max_memory = get_max_memory()
        print("max_memory:", max_memory)
        
        device_map = infer_auto_device_map(model, max_memory=max_memory)
        print("Device map:", device_map)
        
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