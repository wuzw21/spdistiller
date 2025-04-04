import os
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
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(project_root)
from utils.sparse_hook import prepare_sparse_hook
from utils.models import get_llm,get_auto_batch_size
from quantization.qlinear import quant_and_dequant_model_q4_0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument("--task", nargs='+', help="List of tasks to evaluate")
    parser.add_argument('--limit', type=int, default=-1, help='Number of test samples.')
    parser.add_argument('--num_shot',type=int, default=0,help='NUM_SHOT')
    parser.add_argument('--sparse', type=float, default=0,help='sparse')
    parser.add_argument('--do_cr',type=int, default=0,help='cross_layer')
    parser.add_argument('--file_path',type=str, default=0,help='file_path')
    parser.add_argument('--sparse_strategy',type=str, help='sparse strategy')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of test samples.')
    parser.add_argument('--test_all',type=int,default=0)
    args = parser.parse_args()
    return args

def download_dataset(task_name):
    import datasets
    datasets.load_dataset(
        path=task_name,
        split="validation",
        name=None,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        trust_remote_code=True
    )

def eval(
    model,
    tokenizer,
    task_list=["boolq", "rte", "hellaswag", "winogrande", "arc_challenge", "arc_easy", "openbookqa"],
    num_shots=0,
    batch_size=1,
    limit=None
):
    #from .llm_eval_adaptor import LMEvalAdaptor
    from lm_eval import evaluator
    from lm_eval.utils import make_table
    from lm_eval.tasks import TaskManager
    from lm_eval.models.huggingface import HFLM
    task_manager = TaskManager()
    task_names = task_manager.match_tasks(task_list)
    task_num_fewshot = {
        "agieval": 0,
        "mmlu": 5,
        "arc_challenge": 25,
        # 其他任务的 num_fewshot 设置
    }
    lm_eval_model = HFLM(model, tokenizer=tokenizer, batch_size=batch_size, max_length=2048)
    results = []
    for task_name in task_names:
        num_fewshot = task_num_fewshot.get(task_name, num_shots)

        # 评估当前任务
        try:
        # Evaluate the current task
            task_result = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                use_cache=None,
                limit=limit,
                check_integrity=False,
                log_samples=True,
            )
        except Exception as e:
            print(f"Task '{task_name}' testing failed: {e}")

        else :
            print('finish_task', task_name)
            print(make_table(task_result))
            results.append(make_table(task_result))

    # 生成结果表格
    return results

def process_compressed_pred(compressed_pred, output_file, layer_idx, weight_idx):
    """
    处理压缩后的预测结果，提取前 20% 的最大元素，并将结果写入文件。
    
    参数:
        compressed_pred: 压缩后的预测张量 (形状为 (batch_size, hidden_size))
        output_file: 输出文件路径
        layer_idx: 当前层的索引
        weight_idx: 当前权重的索引
    """
    # 按照大小排序
    sorted_pred, indices = torch.sort(compressed_pred, descending=True)
    
    # 提取前 20% 的最大元素
    top_k = int(0.2 * compressed_pred.numel())
    top_elements = sorted_pred[:top_k]
    
    # 计算这些元素占全部元素的比例
    total_sum = compressed_pred.sum().item()
    top_sum = top_elements.sum().item()
    top_ratio = top_sum / total_sum if total_sum > 0 else 0
    
    # 将结果写入文件
    with open(output_file, 'a') as f:
        f.write(f"Layer {layer_idx}, Weight {weight_idx}:\n")
        f.write(f"Top 20% elements: {top_elements.tolist()}\n")
        f.write(f"Top 20% ratio: {top_ratio:.4f}\n")
        f.write(f"Total sum: {total_sum:.4f}\n")
        f.write(f"Top sum: {top_sum:.4f}\n")
        f.write("-" * 40 + "\n")
    
    # 打印结果
    print(f"Layer {layer_idx}, Weight {weight_idx}:")
    # print(f"Top 20% elements: {top_elements.tolist()}")
    print(f"Top 20% ratio: {top_ratio:.4f}")
    print(f"Total sum: {total_sum:.4f}")
    print(f"Top sum: {top_sum:.4f}")
    print("-" * 40)

def debug_test(model):
    global_weight_preditor = model.predictor
    if os.environ.get('DEBUG_CROSSLAYER','0') != '0' :
        print(f"Total elements: {global_weight_preditor.sparse_params[0]}")
        print(f"Zero elements: {global_weight_preditor.sparse_params[1]}")
        print(f"Zero ratio: {global_weight_preditor.sparse_params[1] / global_weight_preditor.sparse_params[0]:.4f}")
        with open('test_crosslayer.txt', 'w') as f:
            pass  # 清空文件内容

        # # 初始化一个字典来存储每层每个 iweight 的相似性结果
        # layer_similarity_results = {iweight: [] for iweight in range(7)}

        # # 遍历所有层和权重
        # for ilayer in range(32):
        #     for iweight in range(7):
        #         similarities = global_weight_preditor.similarity_results[ilayer][iweight]
        #         if similarities:  # 检查列表是否为空
        #             with open('test_crosslayer.txt', 'a') as f:
        #                 f.write(f"Layer {ilayer} Block {iweight} similarity {similarities}\n")
        #             mean = sum(similarities) / len(similarities)
        #             print(f"Layer {ilayer}, Weight {iweight}: Mean Similarity = {mean:.4f}")
        #             layer_similarity_results[iweight].append(mean)  # 将均值存储到对应 iweight 的列表中
        #         else:
        #             print(f"Layer {ilayer}, Weight {iweight}: No similarity data available.")

        # # 计算每个 iweight 的整体均值
        # for iweight, means in layer_similarity_results.items():
        #     if means:  # 检查是否有数据
        #         overall_mean = sum(means) / len(means)
        #         print(f"Overall Mean Similarity for iweight {iweight}: {overall_mean:.4f}")
        #         with open('test_crosslayer.txt', 'a') as f:
        #             f.write(f"Overall Mean Similarity for iweight {iweight}: {overall_mean:.4f} {overall_mean}\n" )
        #     else:
        #         print(f"No data available for iweight {iweight}.")

        # output_file = "top_elements.txt"
        # with open('top_elements.txt', 'w') as f:
        #     pass  # 清空文件内容
        # for ilayer in range(32) :
        #     for iweight in range(7) :
        #         process_compressed_pred(global_weight_preditor.wmetrics[ilayer][iweight], output_file, ilayer, iweight)

def eval_for_sp_config(model_path, model, tokenizer, task_list, num_shot, batch_size, limit, sp_config, file_path, strategy):
    print("sp_config: ", sp_config, "num_shot: ", num_shot)
    global_weight_preditor = model.predictor
    global_weight_preditor.reset()
    # global_weight_preditor = None
    if global_weight_preditor is not None:
        attn_sp, mlp_sp, w_p, do_cr = sp_config
        global_weight_preditor.set_sp_config(attn_sp, mlp_sp, w_p)
        global_weight_preditor.set_do_pre_prediction(do_cr)
    global_weight_preditor.set_sparsity_threshold(file_path)
    global_weight_preditor.set_sparsity_strategy(strategy)


    results = eval(
        model, tokenizer, task_list,
        num_shot, batch_size, limit
    )
    print('='*40)
    print("sp_config: ", sp_config, "num_shot: ", num_shot)
    print("results")
    for result in results :
        print(result)
    print('='*40)
    debug_test(model)
    


def main():
    args = parse_args()

    #download_dataset(args.task)
    #return

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"loading llm model {args.model}")
    model = get_llm(args.model)
    prepare_sparse_hook(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    quant_and_dequant_model_q4_0(model)

    print("task: eval")

    #task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
    #task_list = ["boolq"]
    #task_name = os.environ["EVAL_DATASET"]
    
    print(f"task_name: {args.task}")
    
    limit = None if ( args.limit is None or args.limit < 0 ) else args.limit
    
    batch_size = args.batch_size
    
    task_list = [
        *args.task,
        "wikitext",
        "agieval",      # 确认是否应为 "agi_eval" 或其他名称
        "arc_easy",
        "arc_challenge",
        "piqa",
        "gsm8k",
        
        # "humaneval",
    ]
    if os.environ.get('EASY_TEST','0') == '1':
        task_list = [
            "wikitext",
            "piqa",
            "agieval",      # 确认是否应为 "agi_eval" 或其他名称
            "arc_easy",
            "arc_challenge",
            "gsm8k"
        ]
    print('task_list',task_list)
    sp_configs = [(args.sparse, args.sparse, 0.00, args.do_cr)]
    if args.test_all:
        sp_configs = [(0,0,0,0), (0.5,0.5,0,0), (0.6,0.6,0,0), (args.sparse, args.sparse, 0.00, args.do_cr)]
        sp_configs = list(set(sp_configs))
        
    num_shots= [5]
    
    print('sp_configs: ',sp_configs)
    for sp_config in sp_configs:
        for num_shot in num_shots: 
            if args.test_all and args.sparse_strategy == 'Static':
                model_name = os.environ.get('MODEL_NAME')
                sparse = sp_config[0]
                filepath = f'../threshold/{model_name}/sparse-{sparse}.json'
            else :
                filepath = args.file_path
            eval_for_sp_config(args.model, model, tokenizer, task_list, num_shot, batch_size, limit, sp_config, filepath, args.sparse_strategy)


if __name__ == "__main__":
    main()