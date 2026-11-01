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

## 📚 Project Documentation & Markdown Reports

All analytical explanations, benchmark findings, and theoretical guides are documented in Markdown files:

| Document | Description |
| :--- | :--- |
| [terminal_output_analysis.md](file:///c:/Users/E%20N%20V%20Y/OneDrive%20-%20Ecole%20Centrale%20Casablanca/Bureau/Lab_project/terminal_output_analysis.md) | **Terminal Output Guide:** Plain-English translation of all terminal output, demystifying HuggingFace warnings and explaining the `Ans kept: False` mystery. |
| [hotpotqa_benchmark_report.md](file:///c:/Users/E%20N%20V%20Y/OneDrive%20-%20Ecole%20Centrale%20Casablanca/Bureau/Lab_project/hotpotqa_benchmark_report.md) | **Latest Benchmark Report:** Multi-document RAG compression analytics on HotpotQA, showing the "Distractor Token Waste" problem and the two-stage adaptive architecture. |
| [prompt_compression_guide.md](file:///c:/Users/E%20N%20V%20Y/OneDrive%20-%20Ecole%20Centrale%20Casablanca/Bureau/Lab_project/prompt_compression_guide.md) | Comprehensive foundations on Information Theory, Perplexity, LLMLingua-1, and LLMLingua-2. |
| [llmlingua_1_explained.md](file:///c:/Users/E%20N%20V%20Y/OneDrive%20-%20Ecole%20Centrale%20Casablanca/Bureau/Lab_project/llmlingua_1_explained.md) | In-depth breakdown of LLMLingua-1 (EMNLP 2023) budget controllers & iterative causal perplexity pruning. |
| [llmlingua_2_explained.md](file:///c:/Users/E%20N%20V%20Y/OneDrive%20-%20Ecole%20Centrale%20Casablanca/Bureau/Lab_project/llmlingua_2_explained.md) | In-depth breakdown of LLMLingua-2 (ACL Findings 2024) data distillation, bidirectional classification, and inference. |

---

## 🔬 Experimental Code & Benchmarks

| Script | Purpose | Output Data |
| :--- | :--- | :--- |
| [test_setup.py](file:///c:/Users/E%20N%20V%20Y/OneDrive%20-%20Ecole%20Centrale%20Casablanca/Bureau/Lab_project/test_setup.py) | Verifies CUDA GPU acceleration (MX150), PyTorch, Hugging Face cache on Drive E:. | Console Output |
| [test_llmlingua_compression.py](file:///c:/Users/E%20N%20V%20Y/OneDrive%20-%20Ecole%20Centrale%20Casablanca/Bureau/Lab_project/test_llmlingua_compression.py) | Initial compression test measuring latency and token throughput across ratios. | [first_llmlingua_test_results.json](file:///c:/Users/E%20N%20V%20Y/OneDrive%20-%20Ecole%20Centrale%20Casablanca/Bureau/Lab_project/first_llmlingua_test_results.json) |
| [benchmark_hotpotqa.py](file:///c:/Users/E%20N%20V%20Y/OneDrive%20-%20Ecole%20Centrale%20Casablanca/Bureau/Lab_project/benchmark_hotpotqa.py) | Full RAG multi-document benchmark on HotpotQA measuring answer recall & evidence retention. | [hotpotqa_benchmark_results.json](file:///c:/Users/E%20N%20V%20Y/OneDrive%20-%20Ecole%20Centrale%20Casablanca/Bureau/Lab_project/hotpotqa_benchmark_results.json) |

---

## ⚙️ Environment Setup

- **GPU:** NVIDIA GeForce MX150 (4 GB VRAM)
- **HuggingFace Cache:** `E:\AI_Lab_Environment\hf_cache` (redirected to protect C: drive)
- **Activation Script:**
  ```powershell
  .\activate.ps1
  ```

