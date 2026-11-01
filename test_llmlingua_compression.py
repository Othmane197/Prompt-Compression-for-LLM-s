import os
import sys
import time
import json

# Ensure HF Cache points to drive E:
os.environ["HF_HOME"] = r"E:\AI_Lab_Environment\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"E:\AI_Lab_Environment\hf_cache\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"E:\AI_Lab_Environment\hf_cache\hub"

import torch
from llmlingua import PromptCompressor

def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    print("\n--- Initializing LLMLingua-2 Compressor ---")
    model_name = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    print(f"Loading model: {model_name}...")
    
    start_load = time.time()
    compressor = PromptCompressor(
        model_name=model_name,
        use_llmlingua2=True,
        device_map=device
    )
    print(f"Model loaded in {time.time() - start_load:.2f}s")

    # Sample realistic context: RAG / Document QA scenario
    sample_context = [
        """Large Language Models (LLMs) have achieved remarkable capabilities across a wide range of natural language processing tasks, including question answering, summarization, code generation, and multi-step reasoning. However, as applications become more sophisticated, prompts fed to these models have grown significantly in length. Modern architectures frequently incorporate in-context learning demonstrations, chain-of-thought exemplars, retrieved passages from large knowledge bases, and multi-turn dialogue histories. While long prompts enhance task performance and contextual grounding, they introduce substantial computational bottlenecks: quadratic self-attention scaling, increased time-to-first-token (TTFT) latency, high Key-Value (KV) cache memory overhead, and steep monetary API charges."""
    ]
    
    instruction = "Summarize the key trade-offs of using long prompts in Large Language Models."
    question = "What are the primary computational bottlenecks caused by long prompts?"

    target_rates = [0.8, 0.6, 0.4, 0.25, 0.15] # 1.25x, 1.67x, 2.5x, 4x, 6.67x

    results = []

    print("\n" + "=" * 80)
    print("LLMLINGUA PROMPT COMPRESSION TEST BENCHMARK")
    print("=" * 80)

    full_original_prompt = f"Instruction: {instruction}\nContext: {sample_context[0]}\nQuestion: {question}"
    print(f"\n[ORIGINAL PROMPT PREVIEW]:\n{full_original_prompt}")
    print("=" * 80)

    for rate in target_rates:
        t0 = time.time()
        res = compressor.compress_prompt(
            context=sample_context,
            instruction=instruction,
            question=question,
            rate=rate,
            use_token_level_filter=True
        )
        elapsed = time.time() - t0

        orig_tokens = res["origin_tokens"]
        comp_tokens = res["compressed_tokens"]
        actual_rate = comp_tokens / orig_tokens if orig_tokens > 0 else 0
        actual_ratio = orig_tokens / comp_tokens if comp_tokens > 0 else 0
        token_reduction_pct = (1.0 - actual_rate) * 100
        throughput = orig_tokens / elapsed if elapsed > 0 else 0

        item = {
            "target_rate": rate,
            "target_ratio": f"{1.0/rate:.2f}x",
            "original_tokens": orig_tokens,
            "compressed_tokens": comp_tokens,
            "actual_rate": round(actual_rate, 4),
            "actual_ratio": f"{actual_ratio:.2f}x",
            "tokens_saved": orig_tokens - comp_tokens,
            "reduction_pct": f"{token_reduction_pct:.1f}%",
            "latency_ms": round(elapsed * 1000, 2),
            "throughput_tokens_per_sec": round(throughput, 1),
            "compressed_text": res["compressed_prompt"]
        }
        results.append(item)

        print(f"\n>>> Target Rate: {rate} ({item['target_ratio']} target compression)")
        print(f"    - Retained Tokens: {comp_tokens} / {orig_tokens} ({item['reduction_pct']} saved)")
        print(f"    - Actual Ratio:    {item['actual_ratio']}")
        print(f"    - Latency:         {item['latency_ms']} ms ({item['throughput_tokens_per_sec']} tokens/s)")
        print(f"    - Compressed Text:\n    \"{res['compressed_prompt']}\"")
        print("-" * 80)

    # Save results to a json file for logging & analytics
    with open("first_llmlingua_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nBenchmark completed successfully! Results saved to first_llmlingua_test_results.json")

if __name__ == "__main__":
    run_test()
