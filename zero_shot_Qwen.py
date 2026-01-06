#!/usr/bin/env python3
"""
Zero-shot evaluation script for GSM8K using Qwen2.5-7B with Speed Benchmarking
(Optimized for Robustness and Metric Clarity)
"""

import json
import os
import time
import re
from pathlib import Path
from vllm import LLM, SamplingParams

# --- 配置 ---
# 强制离线模式，防止联网报错
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "./hf_cache"

# 模型 ID
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

DATA_PATH = "data/gsm8k/test.jsonl"
OUTPUT_PATH = "outputs/zero_shot_results.jsonl"

# --- Qwen Prompt ---
QWEN_TEMPLATE = """<|im_start|>system
You are a helpful assistant. Solve the math word problem step by step.
At the end of your solution, you MUST output the final answer in this exact format:
#### [Answer]
For example: #### 42<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

# Sampling parameters
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 1024
STOP_STRINGS = ["<|im_end|>"]


def load_gsm8k_data(data_path):
    examples = []
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return []
    with open(data_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def format_prompts(examples):
    prompts = []
    for ex in examples:
        prompt = QWEN_TEMPLATE.replace("{question}", ex["question"])
        prompts.append(prompt)
    return prompts

def clean_answer(text):
    """
    从文本中提取数值。
    优化了正则逻辑，防止提取到非数值字符。
    """
    if not text:
        return None
    # 移除逗号 (1,000 -> 1000)
    text = text.replace(',', '')
    # 移除末尾的句号 (42. -> 42)
    text = text.rstrip('.')
    
    # 提取所有数字（支持负数和小数）
    # 解释: -? (可选负号) \d+ (整数部分) (\.\d+)? (可选小数部分)
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    
    if matches:
        try:
            # 通常在 GSM8K 中，如果是一段长文本，答案是最后一个数字
            # 但如果是从 #### 提取的短文本，这也能正常工作
            return float(matches[-1])
        except:
            return None
    return None

def evaluate_responses(examples, outputs):
    """
    内置的 GSM8K 评分器，不依赖外部库
    """
    results = []
    
    print("\n" + "="*40)
    print("DEBUG: Checking first 3 responses")
    print("="*40)

    for i, (ex, output) in enumerate(zip(examples, outputs)):
        response = output.outputs[0].text
        
        # 1. 获取标准答案 (Ground Truth)
        gt_str = ex.get("answer", "").split("####")[-1].strip()
        gt_val = clean_answer(gt_str)
        
        # 2. 获取模型预测 (Prediction)
        pred_val = None
        has_format = False
        
        if "####" in response:
            has_format = True
            # [修正点] 提取 #### 后面的内容
            # 这里如果不 strip，后续 extract 可能会被换行符干扰
            pred_str = response.split("####")[-1].strip()
            
            # [修正点] 如果模型在 #### 42 后面还废话，我们尽量取 #### 后面的第一个有效数字
            # 为了安全，我们先尝试只取第一行（通常答案只有一行）
            pred_str_first_line = pred_str.split('\n')[0]
            pred_val = clean_answer(pred_str_first_line)
            
            # 如果第一行没提取到（极少见），再退化到原来的逻辑
            if pred_val is None:
                pred_val = clean_answer(pred_str)
        else:
            # 如果没按格式输出，尝试从最后一段文本提取数字作为补救
            pred_val = clean_answer(response)

        # 3. 比较
        is_correct = False
        if gt_val is not None and pred_val is not None:
            # 浮点数比较，容忍极小误差
            if abs(gt_val - pred_val) < 1e-6:
                is_correct = True
        
        # DEBUG 打印前几个
        if i < 3:
            print(f"\n[Example {i+1}]")
            print(f"GT String: {gt_str} -> Val: {gt_val}")
            print(f"Model Output (Last 100 chars): ...{response[-100:].replace(chr(10), ' ')}")
            print(f"Extracted: {pred_val}")
            print(f"Correct: {is_correct}")

        result = {
            "question": ex["question"],
            "ground_truth": gt_str,
            "model_response": response,
            "format_reward": 1.0 if has_format else 0.0,
            "answer_reward": 1.0 if is_correct else 0.0,
            "reward": 1.0 if is_correct else 0.0
        }
        results.append(result)
    
    return results

def calculate_metrics(results):
    total = len(results)
    if total == 0: return {}
        
    format_correct = sum(1 for r in results if r["format_reward"] == 1.0)
    answer_correct = sum(1 for r in results if r["answer_reward"] == 1.0)
    
    metrics = {
        "total_examples": total,
        "format_accuracy": format_correct / total,
        "answer_accuracy": answer_correct / total,
        "overall_accuracy": answer_correct / total, 
    }
    return metrics

def main():
    print("=" * 60)
    print(f"Zero-shot GSM8K Evaluation with Speed Test (Native Grader)")
    print("=" * 60)
    
    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    examples = load_gsm8k_data(DATA_PATH)
    if not examples: return
    print(f"Loaded {len(examples)} examples")
    
    prompts = format_prompts(examples)
    
    # [修正点] 显式设置 GPU 利用率，防止碎片化导致 OOM
    print(f"Initializing vLLM: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME, 
        trust_remote_code=True,
        gpu_memory_utilization=0.9, # 建议设置，利用率更高
        max_model_len=4096 # 限制一下最大长度防止显存溢出
    )
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stop=STOP_STRINGS,
        include_stop_str_in_output=False 
    )
    
    print(f"Generating responses...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    # [修正点] 更详细的 Metrics 统计
    total_duration = end_time - start_time
    total_output_tokens = sum([len(o.outputs[0].token_ids) for o in outputs])
    total_prompt_tokens = sum([len(o.prompt_token_ids) for o in outputs])
    total_tokens = total_output_tokens + total_prompt_tokens
    
    # 计算真正的系统吞吐量
    system_throughput = total_tokens / total_duration if total_duration > 0 else 0
    output_throughput = total_output_tokens / total_duration if total_duration > 0 else 0
    
    print("-" * 30)
    print(f"BENCHMARK RESULTS (A100 Throughput)")
    print(f"Total Duration:     {total_duration:.2f} s")
    print(f"Total Requests:     {len(outputs)}")
    print(f"Avg Output Length:  {total_output_tokens / len(outputs):.1f} tokens")
    print("-" * 30)
    print(f"System Throughput:  {system_throughput:.2f} tokens/sec (Prefill + Decode)")
    print(f"Output Throughput:  {output_throughput:.2f} tokens/sec (Effective Generation)")
    print(f"* Note: 'System Throughput' explains the high 4800+ number you saw.")
    print("-" * 30)
    
    results = evaluate_responses(examples, outputs)
    metrics = calculate_metrics(results)
    
    print(f"\n[SAVE] Writing results to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    if metrics:
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        print(f"Format Accuracy: {metrics['format_accuracy']:.2%}")
        print(f"Answer Accuracy: {metrics['answer_accuracy']:.2%}")
        print("=" * 60)

if __name__ == "__main__":
    main()