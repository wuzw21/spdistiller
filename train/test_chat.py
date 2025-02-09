from transformers import AutoModelForCausalLM,AutoTokenizer
model_path = '/data/wzw/models/Meta-Llama-3-8B'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="balanced",
        trust_remote_code=True
    )
print(model)
while(True) :
    i = input('>')
    inputs = tokenizer(i, return_tensors="pt").to(model.device)

    # 模型生成输出
    outputs = model.generate(inputs['input_ids'], max_length=100, do_sample=True)

    # 解码输出张量为文本
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)