# Adaptive Prompt Compression for Efficient Large Language Model Inference

**Academic Lab Project** — CentraleSupélec / Manifold Technology  
**Supervisor:** Yannick LE CACHEUX (`yannick.le-cacheux@centralesupelec.fr`)  

---

## 📌 Project Overview

Large Language Models (LLMs) increasingly rely on long prompts containing retrieved documents (RAG), few-shot demonstrations (ICL), and conversational histories. These extended contexts introduce severe bottlenecks:
1. **High API & compute costs** (billed per token).
2. **Inference latency** (attention scaling & Time-To-First-Token delay).
3. **The "Lost-in-the-Middle" degradation** where LLMs lose focus across noisy text.

This project investigates **Adaptive Prompt Compression**: dynamically determining which segments of a heterogeneous prompt require gentle preservation and which segments tolerate aggressive pruning.

---

## 📚 Documentation Index (`docs/`)

All theoretical guides, benchmark reports, and replication analyses are organized under `docs/`:

### 🔬 Core Methodology & Theory (`docs/guides/` & `docs/methodologies/`)
- [fixed_vs_dynamic_ratios.md](docs/methodologies/fixed_vs_dynamic_ratios.md): **Budget Dynamics:** Explains fixed vs. dynamic non-uniform compression ratios across LLMLingua-1, LongLLMLingua, and LLMLingua-2.
- [adaptive_compression_strategies.md](docs/methodologies/adaptive_compression_strategies.md): **Core Thesis Roadmap:** Why uniform compression fails, how LongLLMLingua/LLMLingua-1 distributed budgets, and our proposed innovations (Two-Stage Adaptive RAG, Structure-Agnostic Sensitivity, Query-Conditioned Pruning).
- [prompt_compression_guide.md](docs/guides/prompt_compression_guide.md): Information Theory foundations, perplexity, and compression fundamentals.
- [llmlingua_1_explained.md](docs/guides/llmlingua_1_explained.md): Budget controller & iterative causal perplexity pruning (EMNLP 2023).
- [llmlingua_2_explained.md](docs/guides/llmlingua_2_explained.md): Data distillation, bidirectional token classification, subword pooling, and inference (ACL Findings 2024).

### 📊 Benchmark Reports (`docs/benchmarks/`)
- [experiments_log.md](docs/benchmarks/experiments_log.md): Complete chronological log of all tests conducted, results tables, and copy-paste run commands.
- [hotpotqa_benchmark_report.md](docs/benchmarks/hotpotqa_benchmark_report.md): Multi-document RAG compression benchmark results across 4 compression rates on NVIDIA MX150.
- [hotpotqa_evaluation_guide.md](docs/benchmarks/hotpotqa_evaluation_guide.md): Guide to gold evidence metrics, bridge vs. comparison question types, and entity fragility.

### 🔄 Independent Replication Study (`docs/replication/`)
- [peer_colab_replication_study.md](docs/replication/peer_colab_replication_study.md): MeetingBank replication analysis on Mistral-7B, pipeline mapping, and comparative insights.

---

## 💻 Experimental Scripts & Data

| Script / Data | Description |
| :--- | :--- |
| `benchmark_hotpotqa.py` | Multi-document RAG benchmark testing LLMLingua-2 on HotpotQA across compression rates. |
| `test_llmlingua_compression.py` | Initial speed, latency, and throughput verification script. |
| `test_setup.py` | Environment and GPU sanity check script. |
| `hotpotqa_benchmark_results.json` | Full quantitative evaluation output across 40 test runs. |
| `first_llmlingua_test_results.json` | Raw output trace of initial single-prompt benchmark. |

---

## ⚙️ Environment Setup

- **GPU:** NVIDIA GeForce MX150 (4 GB VRAM)
- **HuggingFace Cache:** `E:\AI_Lab_Environment\hf_cache` (redirected to protect C: drive)
- **Activation Script:**
  ```powershell
  .\activate.ps1
  ```
