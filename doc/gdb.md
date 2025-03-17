Step 1: 在Apple上Push debug版本的mymain到手机
adb -s xxx push mymain-tg-aarch64 /data/local/tmp/elastic-infer
Step 2: 在Apple上映射debug端口
adb -s xxx forward tcp:5039 tcp:5039
Step 3: 在手机上启动debug mymain
adb -s xxx shell

cd /data/local/tmp/elastic-infer

export LLAMA_COMPUTING_THREAD_CPU_CORE_IDS=4,5,6,7
export LLAMA_LOADING_THREAD_CPU_CORE_ID=0

export LLAMA_LOAD_SPARSE_TENSOR=1
export LLAMA_USE_OUT_DIM_INDEX_CACHE=1

LD_LIBRARY_PATH=/data/local/tmp/elastic-infer ./gdbserver :5039 ./mymain-tg-aarch64 -ngl 999 -m /data/local/tmp/elastic-infer/Llama-2-7b-chat-hf-t-Q4_0.gguf -n 4 -p \"Once upon a\" --temp -1.0 --attn-sp 0.50 --ffn-sp 0.00 -b 1
LD_LIBRARY_PATH=/data/local/tmp/elastic-infer ./mymain-t-aarch64 -ngl 999 -m /data/local/tmp/elastic-infer/Llama-2-7b-chat-hf-t-Q4_0.gguf -n 10 -p "Once upon a" --temp -1.0 --attn-sp 0.50 --ffn-sp 0.50 -b 1 -t 4
注：gdbserver程序在/data/local/tmp/v-fuchengjia下面有，直接copy即可

Step 4: 在Apple上启动GDB
cd ~/fuchengjia/Downloads/android-ndk-r21e/prebuilt/darwin-x86_64/bin

./gdb
Step 5: 链接到mymain
target remote 127.0.0.1:5039

Press C to continue

可能出现的segment fault: su