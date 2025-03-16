账号：
密码：android@shanghai

Step 0: Prepare
Step 0-0: Create a directory for testing on your smartphone
adb shell mkdir -p /data/local/tmp/elastic-infer
Step 0-1: ADB pushing files
# Libraries.
adb push libc++_shared.so /data/local/tmp/elastic-infer
adb push liburing.so.2.8 /data/local/tmp/elastic-infer

# Executables.
adb push mymain-nt-aarch64 /data/local/tmp/elastic-infer
adb push mymain-t-aarch64 /data/local/tmp/elastic-infer

# Models.
# NOTE: nt=not transpose for weights, t=transpose for weights
adb push Llama-2-7b-chat-hf-nt-Q4_0.gguf /data/local/tmp/elastic-infer
adb push Llama-2-7b-chat-hf-t-Q4_0.gguf /data/local/tmp/elastic-infer
Step 1: Set shell environments
# Enter directory.
adb shell
cd /data/local/tmp/elastic-infer

# Library path.
export LD_LIBRARY_PATH=/data/local/tmp/elastic-infer:${LD_LIBRARY_PATH}

# Bind cpu cores.
# 8 cores for 8 threads.
export LLAMA_COMPUTING_THREAD_CPU_CORE_IDS=0,1,2,3,4,5,6,7
export LLAMA_LOADING_THREAD_CPU_CORE_ID=0

# 6 cores for 6 threads.
export LLAMA_COMPUTING_THREAD_CPU_CORE_IDS=2,3,4,5,6,7
export LLAMA_LOADING_THREAD_CPU_CORE_ID=0

# 4 cores for 4 threads.
export LLAMA_COMPUTING_THREAD_CPU_CORE_IDS=4,5,6,7
export LLAMA_LOADING_THREAD_CPU_CORE_ID=0
Step 2: Test llama.cpp baseline
# Run.
# NOTE: -m=model_file, -n=max_num_generate_tokens, -p=promt_text, --temp=temperature_for_sampling, --attn-sp=sparsity of attention, --ffn-sp=sparsity of FFN, -b=batch, -t=num_threads
./mymain-nt-aarch64 -ngl 999 -m ./Llama-2-7b-chat-hf-nt-Q4_0.gguf -n 4 -p "Once upon a time"  --temp -1.0 --attn-sp 0.00 --ffn-sp 0.00 -b 1 -t 8
The encoding and decoding latency (ms/token) and speed (tokens/sec) are reported as follow:

# Encoding.
llama_print_timings: prompt eval time =    3010.72 ms /     5 tokens (  602.14 ms per token,     1.66 tokens per second)
# Decoding.
llama_print_timings:        eval time =    1839.57 ms /     3 runs   (  613.19 ms per token,     1.63 tokens per second)
Step 3: Test our pipeline
# Enable weight loading.
export LLAMA_LOAD_SPARSE_TENSOR=1

# Enable caching.
export LLAMA_USE_OUT_DIM_INDEX_CACHE=1

# Run with 50% sparsity for decoding.
./mymain-t-aarch64 -ngl 999 -m ./Llama-2-7b-chat-hf-t-Q4_0.gguf -n 4 -p "Once upon a time" --temp -1.0 --attn-sp 0.50 --ffn-sp 0.50 -b 1 -t 8

# Run with 70% sparsity for decoding.
./mymain-t-aarch64 -ngl 999 -m ./Llama-2-7b-chat-hf-t-Q4_0.gguf -n 4 -p "Once upon a time" --temp -1.0 --attn-sp 0.70 --ffn-sp 0.70 -b 1 -t 8

# Run with 80% sparsity for decoding.
./mymain-t-aarch64 -ngl 999 -m ./Llama-2-7b-chat-hf-t-Q4_0.gguf -n 4 -p "Once upon a time" --temp -1.0 --attn-sp 0.80 --ffn-sp 0.80 -b 1 -t 8
Step 4: Analyze performance bound by testing
Step 4-1: Synchronization-bound or not?
# Disable weight loading.
export LLAMA_LOAD_SPARSE_TENSOR=0

# Run with 50% sparsity for decoding.
./mymain-t-aarch64 -ngl 999 -m ./Llama-2-7b-chat-hf-t-Q4_0.gguf -n 4 -p "Once upon a time" --temp -1.0 --attn-sp 0.50 --ffn-sp 0.50 -b 1 -t 8
We get the encoding time and decoding time marked as $T_{enc}$ and $T_{dec}$. * If $T_{enc} << T_{dec}$, it is bounded by synchronization between computing thread and loading thread, because we only do synchronize for decoding. * If $T_{enc} \approx T_{dec}$, go to step 4-2.

Step 4-2: Flash-IO-bound or RAM-IO-bound?
# Enable weight loading and caching.
export LLAMA_LOAD_SPARSE_TENSOR=1
export LLAMA_USE_OUT_DIM_INDEX_CACHE=1

# Run with 50% sparsity for decoding.
./mymain-t-aarch64 -ngl 999 -m ./Llama-2-7b-chat-hf-t-Q4_0.gguf -n 4 -p "Once upon a time" --temp -1.0 --attn-sp 0.50 --ffn-sp 0.50 -b 1 -t 8
We get the encoding time and decoding time marked as $T^{'}_{enc}$ and $T^{'}_{dec}$. * If $T^{'}_{dec} >> T_{dec}$, it is Flash-IO-bound. * If $T^{'}_{dec} \approx T_{dec}$, it is RAM-IO-bound.

TODO
Now we use num_cross_layers=4 as default. However, it is an important factor for IO weight loading performance and latency. We will add this argument for flexible testing.