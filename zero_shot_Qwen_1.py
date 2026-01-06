#!/usr/bin/env python3
"""
Single-Stream Latency Benchmark for Qwen2.5-7B
(Measures how fast tokens are generated for a single user)
"""

import time
import os
import numpy as np
from vllm import LLM, SamplingParams

# --- 配置 ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "./hf_cache"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 测试参数
NUM_WARMUP = 3      # 预热次数 (让GPU进入状态)
NUM_TESTS = 10      # 正式测试次数 (取平均值)
GEN_TOKENS = 256    # 每次生成的 token 数量 (足够长以稳定速度)

def main():
    print("=" * 50)
    print(f"Initializing vLLM for Single-Stream Latency Test")
    print(f"Model: {MODEL_NAME}")
    print("=" * 50)

    # 初始化模型
    # 这里的关键是：虽然 vLLM 是为高并发设计的，
    # 但我们只发一个请求，它就会全力以赴处理这一个。
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        tensor_parallel_size=1, 
        gpu_memory_utilization=0.9,
        # 禁用一些可能影响单流测速的高级调度日志
        disable_log_stats=True 
    )

    # 构造采样参数
    # ignore_eos=True 确保模型不会因为生成了 "<|im_end|>" 就提前停止
    # 这样我们可以精确控制生成 GEN_TOKENS 个 token
    sampling_params = SamplingParams(
        temperature=0.0,    # 贪婪采样，最快且确定性最高
        min_tokens=GEN_TOKENS,
        max_tokens=GEN_TOKENS,
        ignore_eos=True
    )

    # 构造一个简单的 Prompt
    # 我们不关心内容，只关心生成过程
    dummy_prompt = "Hello, please output random words to test speed."
    prompts = [dummy_prompt] # 列表里只有1个元素 -> 单流

    print(f"\n[Phase 1] Warming up GPU ({NUM_WARMUP} runs)...")
    for i in range(NUM_WARMUP):
        llm.generate(prompts, sampling_params)
        print(f"  Warmup {i+1}/{NUM_WARMUP} done.")

    print(f"\n[Phase 2] Running Benchmark ({NUM_TESTS} runs)...")
    speeds = []
    
    for i in range(NUM_TESTS):
        # 记录纯生成时间
        # 注意：vLLM 的 generate 包含了 Prefill 和 Decode。
        # 但对于短 Prompt + 长 Output，Decode 占绝大部分时间，
        # 所以 tokens/total_time 是最真实的“用户端感知速度”。
        start_time = time.perf_counter() # 使用高精度计时器
        
        outputs = llm.generate(prompts, sampling_params)
        
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        # 获取实际生成的 token 数 (应该是 GEN_TOKENS)
        num_tokens = len(outputs[0].outputs[0].token_ids)
        
        speed = num_tokens / duration
        speeds.append(speed)
        
        print(f"  Run {i+1:02d}: {speed:.2f} tokens/sec ({num_tokens} tokens in {duration:.4f}s)")

    # 计算统计结果
    avg_speed = np.mean(speeds)
    p99_speed = np.percentile(speeds, 99)
    p01_speed = np.percentile(speeds, 1)

    print("\n" + "=" * 50)
    print(f"FINAL RESULT: Single-Stream Decoding Speed")
    print("=" * 50)
    print(f"Hardware:        A100 (Assumed based on context)")
    print(f"Average Speed:   {avg_speed:.2f} tokens/sec")
    print(f"Min Speed:       {min([s for s in speeds]):.2f} tokens/sec")
    print(f"Max Speed:       {max([s for s in speeds]):.2f} tokens/sec")
    print("-" * 50)
    print("Interpretation:")
    print(f"This is the speed a single user feels when chatting.")
    print(f"Expected range on A100: 130 - 170 tokens/sec")
    print("=" * 50)

if __name__ == "__main__":
    main()