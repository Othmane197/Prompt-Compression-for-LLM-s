# Experiments & Benchmark Log

This document records all experimental tests conducted in this repository, including objectives, execution commands, hardware environment, and empirical results.

---

## 🛠️ Quick Commands to Run the Tests

Open PowerShell in the project directory (`Lab_project`):

### Option A: Standard Workflow (Activate First)
```powershell
# 1. Activate the AI lab environment (sets HF_HOME to Drive E: and loads .venv)
.\activate.ps1

# 2. Run Test 1: Single-Prompt Compression Across 5 Ratios
python test_llmlingua_compression.py

# 3. Run Test 2: HotpotQA Multi-Document RAG Benchmark (reference run: 10 samples, 4 rates)
python benchmark_hotpotqa.py
#    Quick test / custom run (writes hotpotqa_results_n<N>.json, never overwrites the reference results)
python benchmark_hotpotqa.py -n 2 -r 0.5 0.2

# 4. Run Environment Sanity Check
python test_setup.py
```

### Option B: One-Liner Direct Execution (Never Fails)
```powershell
# Direct execution without activating shell session
.\.venv\Scripts\python.exe test_llmlingua_compression.py
.\.venv\Scripts\python.exe benchmark_hotpotqa.py
```

---

## 🔬 Test 0: Environment & Hardware Verification

* **Script:** `test_setup.py`
* **Objective:** Ensure CUDA GPU acceleration works and that all Hugging Face downloads are stored on Drive E: (protecting the low-space C: drive).
* **Hardware Verified:**
  - **GPU:** NVIDIA GeForce MX150 (4.00 GB VRAM)
  - **PyTorch:** `2.5.1+cu121` (CUDA Available: `True`)
  - **Hugging Face Cache:** `E:\AI_Lab_Environment\hf_cache` (580+ GB free space)
  - **NLP Libraries:** `transformers 4.47.1`, `datasets 3.6.0`, `llmlingua 0.2.2`, `spaCy en_core_web_sm`

---

## 🔬 Test 1: Initial Multi-Ratio LLMLingua-2 Benchmark

* **Script:** `test_llmlingua_compression.py`
* **Output Data:** `first_llmlingua_test_results.json`
* **Model:** `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
* **Input Prompt:** 136 tokens (RAG background paragraph + instruction + question)
* **Objective:** Verify LLMLingua-2 inference on GPU and measure latency and throughput across 5 compression rates.

### Empirical Results Table:

| Target Rate ($\tau$) | Target Ratio | Kept Tokens | Tokens Saved | Reduction (%) | Actual Ratio | Inference Latency | Throughput |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.80** | 1.25x | 111 / 136 | 25 | **18.4%** | 1.23x | 18.6 s *(CUDA warmup)* | 7.3 tok/s |
| **0.60** | 1.67x | 77 / 136 | 59 | **43.4%** | 1.77x | **215 ms** | **631.5 tok/s** |
| **0.40** | 2.50x | 56 / 136 | 80 | **58.8%** | 2.43x | **209 ms** | **650.0 tok/s** |
| **0.25** | 4.00x | 32 / 136 | 104 | **76.5%** | 4.25x | **218 ms** | **622.3 tok/s** |
| **0.15** | 6.67x | 20 / 136 | 116 | **85.3%** | 6.80x | **211 ms** | **643.0 tok/s** |

### Key Takeaway:
After the initial CUDA warmup pass (18.6s), steady-state compression runs in **~210 ms** on the MX150 GPU with a throughput of **>630 tokens/second**, proving that compression overhead is negligible.

---

## 🔬 Test 2: HotpotQA Multi-Document RAG Benchmark

* **Script:** `benchmark_hotpotqa.py`
* **Output Data:** `hotpotqa_benchmark_results.json`
* **Detailed Report:** `docs/benchmarks/hotpotqa_benchmark_report.md`
* **Dataset:** HotpotQA Validation Set (Distractor Setting)
* **Scale:** 10 multi-document prompts (**15,076 total prompt tokens**, ~1,500 tokens per prompt across 10 retrieved passages).
* **Objective:** Test out-of-domain transfer on multi-document inputs and evaluate evidence survival vs. distractor noise survival.

### Empirical Results Table:

| Target Rate ($\tau$) | Actual Ratio | Total Kept Tokens | Token Reduction | Avg Latency | Avg Throughput | Extractive Ans Recall | Gold Evidence Retained | Distractor Noise Retained | Selectivity Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.80** (1.25x) | **1.67x** | 9,035 / 15,076 | **40.1%** | 5.83 s | 287.3 tok/s | **100.0%** | **93.2%** | 80.0% | **+13.2%** |
| **0.50** (2.00x) | **3.55x** | 4,242 / 15,076 | **71.9%** | 5.47 s | 314.8 tok/s | **85.7%** | **60.9%** | 48.2% | **+12.7%** |
| **0.33** (3.03x) | **5.41x** | 2,787 / 15,076 | **81.5%** | 5.87 s | 304.4 tok/s | **42.9%** | **44.9%** | 31.6% | **+13.3%** |
| **0.20** (5.00x) | **10.96x** | 1,375 / 15,076 | **90.9%** | 5.35 s | 314.4 tok/s | **14.3%** | **23.6%** | 15.4% | **+8.2%** |

### Key Scientific Discovery:
1. **The "Distractor Token Waste" Problem:**  
   At $\tau = 0.50$, the compressor kept **60.9% of the true Gold Evidence**, but it also retained **48.2% of the irrelevant Distractor noise**. Uniform compression wastes half its budget preserving irrelevant text.
2. **Safe Operating Range:**  
   Extractive answers survive with **100% fidelity at $\tau = 0.80$** (1.67x actual), and already drop to **85.7% at $\tau = 0.50$** (3.55x actual), where the numeric answer `"3,677 seated"` is lost. Beyond that, survival collapses (42.9% → 14.3%), supporting the need for **query-aware adaptive budget allocation**.

