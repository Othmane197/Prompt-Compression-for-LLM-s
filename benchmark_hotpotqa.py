import os
import sys
import time
import json
import re
from typing import List, Dict, Any

# Ensure HF cache is strictly on Drive E:
os.environ["HF_HOME"] = r"E:\AI_Lab_Environment\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"E:\AI_Lab_Environment\hf_cache\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"E:\AI_Lab_Environment\hf_cache\hub"

import torch
import datasets
from llmlingua import PromptCompressor

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_simple(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'\b\w+\b', text)]

def check_answer_presence(answer: str, compressed_text: str) -> bool:
    """Checks if the ground-truth answer tokens are preserved in the compressed text."""
    ans_tokens = tokenize_simple(answer)
    comp_tokens = set(tokenize_simple(compressed_text))
    if not ans_tokens:
        return False
    # If all tokens of the answer appear in the compressed text
    return all(tok in comp_tokens for tok in ans_tokens)

def compute_evidence_retention(gold_sentences: List[str], distractor_sentences: List[str], compressed_text: str) -> Dict[str, float]:
    """Calculates lexical recall of gold supporting evidence vs distractor text in compressed output."""
    comp_tokens = set(tokenize_simple(compressed_text))
    
    gold_tokens = tokenize_simple(" ".join(gold_sentences))
    distractor_tokens = tokenize_simple(" ".join(distractor_sentences))
    
    gold_kept = sum(1 for tok in gold_tokens if tok in comp_tokens)
    gold_recall = (gold_kept / len(gold_tokens)) if gold_tokens else 0.0
    
    dist_kept = sum(1 for tok in distractor_tokens if tok in comp_tokens)
    dist_recall = (dist_kept / len(distractor_tokens)) if distractor_tokens else 0.0
    
    return {
        "gold_retention_rate": round(gold_recall, 4),
        "distractor_retention_rate": round(dist_recall, 4),
        "evidence_selectivity": round(gold_recall - dist_recall, 4)
    }

def run_hotpotqa_benchmark(num_samples: int = 10, rates: List[float] = [0.8, 0.5, 0.33, 0.2]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 85)
    print(f"   HOTPOTQA MULTI-DOCUMENT RAG BENCHMARK FOR PROMPT COMPRESSION")
    print(f"   Device: {device} | Target Rates: {rates}")
    print("=" * 85)
    
    # 1. Load LLMLingua-2 Model
    model_name = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    print(f"\n[1/3] Loading LLMLingua-2 model: {model_name}...")
    t_start = time.time()
    compressor = PromptCompressor(
        model_name=model_name,
        use_llmlingua2=True,
        device_map=device
    )
    print(f"Model loaded successfully in {time.time() - t_start:.2f}s")
    
    # 2. Load HotpotQA Dataset
    print(f"\n[2/3] Loading {num_samples} samples from HotpotQA validation set...")
    dataset = datasets.load_dataset("hotpot_qa", "distractor", split=f"validation[:{num_samples}]")
    print(f"Loaded {len(dataset)} evaluation samples.")
    
    # 3. Benchmark Loop
    print("\n[3/3] Running compression experiments across compression rates...")
    all_results = []
    
    # Summary accumulators per rate
    rate_summaries = {
        r: {
            "total_orig_tokens": 0,
            "total_comp_tokens": 0,
            "latencies_ms": [],
            "throughputs": [],
            "answer_preserved_count": 0,
            "gold_retentions": [],
            "distractor_retentions": [],
            "selectivities": []
        }
        for r in rates
    }

    instruction = "Answer the question based on the provided context passages. Keep your answer concise."

    for idx, sample in enumerate(dataset):
        q_id = sample["id"]
        question = sample["question"]
        answer = sample["answer"]
        
        # Build context documents & separate gold evidence from distractors
        titles = sample["context"]["title"]
        sentence_lists = sample["context"]["sentences"]
        
        gold_pairs = set(zip(sample["supporting_facts"]["title"], sample["supporting_facts"]["sent_id"]))
        
        gold_sentences = []
        distractor_sentences = []
        context_passages = []
        
        for doc_title, sents in zip(titles, sentence_lists):
            doc_body = " ".join(sents)
            formatted_doc = f"Document [{doc_title}]: {doc_body}"
            context_passages.append(formatted_doc)
            
            for s_idx, sent in enumerate(sents):
                if (doc_title, s_idx) in gold_pairs:
                    gold_sentences.append(sent)
                else:
                    distractor_sentences.append(sent)
                    
        print(f"\n-----------------------------------------------------------------------------")
        print(f"Sample {idx+1}/{num_samples} [ID: {q_id}]")
        print(f"Question: \"{question}\"")
        print(f"Ground Truth Answer: \"{answer}\"")
        print(f"Gold Supporting Facts: {len(gold_sentences)} sentences | Distractors: {len(distractor_sentences)} sentences")
        
        sample_eval = {
            "sample_index": idx + 1,
            "id": q_id,
            "question": question,
            "answer": answer,
            "num_passages": len(context_passages),
            "rates_evaluated": {}
        }
        
        for rate in rates:
            t0 = time.time()
            res = compressor.compress_prompt(
                context=context_passages,
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
            latency_ms = elapsed * 1000
            throughput = orig_tokens / elapsed if elapsed > 0 else 0
            
            ans_present = check_answer_presence(answer, res["compressed_prompt"])
            evidence_metrics = compute_evidence_retention(gold_sentences, distractor_sentences, res["compressed_prompt"])
            
            # Accumulate summaries
            stats = rate_summaries[rate]
            stats["total_orig_tokens"] += orig_tokens
            stats["total_comp_tokens"] += comp_tokens
            stats["latencies_ms"].append(latency_ms)
            stats["throughputs"].append(throughput)
            if ans_present:
                stats["answer_preserved_count"] += 1
            stats["gold_retentions"].append(evidence_metrics["gold_retention_rate"])
            stats["distractor_retentions"].append(evidence_metrics["distractor_retention_rate"])
            stats["selectivities"].append(evidence_metrics["evidence_selectivity"])
            
            sample_eval["rates_evaluated"][rate] = {
                "orig_tokens": orig_tokens,
                "comp_tokens": comp_tokens,
                "actual_rate": round(actual_rate, 4),
                "actual_ratio": f"{actual_ratio:.2f}x",
                "reduction_pct": f"{(1 - actual_rate)*100:.1f}%",
                "latency_ms": round(latency_ms, 2),
                "throughput_tok_s": round(throughput, 1),
                "answer_preserved": ans_present,
                "gold_evidence_retention": evidence_metrics["gold_retention_rate"],
                "distractor_retention": evidence_metrics["distractor_retention_rate"],
                "selectivity": evidence_metrics["evidence_selectivity"],
                "compressed_prompt_preview": res["compressed_prompt"][:180] + "..."
            }
            
            print(f"  > Rate {rate:.2f} ({1/rate:.2f}x): {comp_tokens}/{orig_tokens} tok ({(1-actual_rate)*100:.1f}% cut) "
                  f"| {latency_ms:.1f}ms | Ans kept: {ans_present} | Gold Evidence Retained: {evidence_metrics['gold_retention_rate']*100:.1f}%")
            
        all_results.append(sample_eval)
        
    # Compute Aggregate Analytics
    aggregate_report = []
    print("\n" + "=" * 85)
    print("                    HOTPOTQA BENCHMARK AGGREGATE RESULTS")
    print("=" * 85)
    print(f"{'Rate':<6} | {'Ratio':<6} | {'Tokens Kept':<14} | {'Reduction':<10} | {'Latency':<10} | {'Throughput':<14} | {'Ans Recall':<11} | {'Gold Evidence'}")
    print("-" * 85)
    
    for rate in rates:
        s = rate_summaries[rate]
        tot_orig = s["total_orig_tokens"]
        tot_comp = s["total_comp_tokens"]
        overall_actual_rate = tot_comp / tot_orig if tot_orig > 0 else 0
        overall_ratio = tot_orig / tot_comp if tot_comp > 0 else 0
        avg_latency = sum(s["latencies_ms"]) / len(s["latencies_ms"])
        avg_throughput = sum(s["throughputs"]) / len(s["throughputs"])
        ans_recall = (s["answer_preserved_count"] / num_samples) * 100
        avg_gold_retention = (sum(s["gold_retentions"]) / len(s["gold_retentions"])) * 100
        avg_dist_retention = (sum(s["distractor_retentions"]) / len(s["distractor_retentions"])) * 100
        
        row_str = (f"{rate:<6.2f} | {overall_ratio:<5.2f}x | {tot_comp}/{tot_orig} ({overall_actual_rate*100:.1f}%) | "
                   f"{(1-overall_actual_rate)*100:<9.1f}% | {avg_latency:<8.1f}ms | {avg_throughput:<10.1f} tok/s | "
                   f"{ans_recall:<10.1f}% | {avg_gold_retention:.1f}% (vs {avg_dist_retention:.1f}% dist)")
        print(row_str)
        
        aggregate_report.append({
            "target_rate": rate,
            "target_ratio": f"{1/rate:.2f}x",
            "actual_rate": round(overall_actual_rate, 4),
            "actual_ratio": f"{overall_ratio:.2f}x",
            "token_reduction_pct": round((1 - overall_actual_rate) * 100, 2),
            "total_original_tokens": tot_orig,
            "total_compressed_tokens": tot_comp,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_throughput_tok_s": round(avg_throughput, 1),
            "answer_retention_pct": round(ans_recall, 1),
            "avg_gold_evidence_retention_pct": round(avg_gold_retention, 2),
            "avg_distractor_retention_pct": round(avg_dist_retention, 2),
            "average_selectivity_gap": round(avg_gold_retention - avg_dist_retention, 2)
        })

    # Save to JSON
    output_payload = {
        "benchmark_metadata": {
            "dataset": "hotpot_qa (distractor)",
            "split": "validation",
            "num_samples": num_samples,
            "model": model_name,
            "device": device
        },
        "aggregate_summary": aggregate_report,
        "sample_details": all_results
    }
    
    with open("hotpotqa_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
        
    print("\nBenchmark finished! Full data saved to hotpotqa_benchmark_results.json")

if __name__ == "__main__":
    run_hotpotqa_benchmark(num_samples=10, rates=[0.8, 0.5, 0.33, 0.2])

