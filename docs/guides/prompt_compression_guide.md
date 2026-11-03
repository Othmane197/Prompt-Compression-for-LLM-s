# Prompt Compression: Foundations, Methodologies & Lab Project Guide

**Project:** Adaptive Prompt Compression for Large Language Models  
**Reference Papers:**
- **Paper 1:** *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models* (EMNLP 2023) — [2023.emnlp-main.825.pdf](file:///c:/Users/E%20N%20V%20Y/OneDrive%20-%20Ecole%20Centrale%20Casablanca/Bureau/Lab_project/2023.emnlp-main.825.pdf)
- **Paper 2:** *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression* (Findings of ACL 2024) — [2024.findings-acl.57.pdf](file:///c:/Users/E%20N%20V%20Y/OneDrive%20-%20Ecole%20Centrale%20Casablanca/Bureau/Lab_project/2024.findings-acl.57.pdf)

---

## 1. Executive Summary & Core Motivation

### The Problem
Large Language Models (LLMs) are widely deployed with rich context:
- In-Context Learning (ICL) few-shot demonstrations
- Chain-of-Thought (CoT) multi-step reasoning exemplars
- Retrieval-Augmented Generation (RAG) retrieved documents
- Extended conversational histories

However, long prompts create three critical bottlenecks:
1. **High API Costs:** Commercial models (GPT-4, Claude) bill per token for both prompt input and generation.
2. **Inference Latency:** Attention mechanisms and Key-Value (KV) cache loading slow down Time-To-First-Token (TTFT).
3. **Performance Degradation:** The "Lost-in-the-Middle" phenomenon shows that LLMs lose accuracy when critical details are drowned in irrelevant context.

### The Foundation
Both papers build on Claude Shannon’s 1951 Information Theory principle: **Natural language is highly redundant.**  
A significant portion of words in human speech and writing consists of syntactic glue, filler words, or repetitive phrasing. If we remove these redundant tokens, an LLM can still reconstruct the meaning and generate the exact same answer.

### Core Definitions
- **Original Prompt (x):** The raw input with length L tokens.
- **Compressed Prompt (x_tilde):** The shortened prompt with length L_tilde tokens.
- **Compression Rate (tau):**
  $$\tau = \frac{\tilde{L}}{L} \in (0, 1]$$
  *(In plain terms: tau = Compressed Tokens / Original Tokens. For example, tau = 0.20 means retaining 20% of the words).*
- **Compression Ratio:**
  $$\text{Ratio} = \frac{1}{\tau}$$
  *(In plain terms: 1 / tau. For example, if tau = 0.20, the ratio is 1 / 0.20 = 5x compression).*
- **Black-Box Setup:** The target LLM (e.g., GPT-4) is fixed and accessed via an API. A small, cheap model performs the compression locally before calling the API.

---

## 2. Paper 1: LLMLingua (EMNLP 2023)

### 2.1 The Core Idea: Perplexity as Informativeness
LLMLingua uses a small causal language model (such as LLaMA-7B or GPT-2) to measure how "surprised" it is by each token in the prompt:
- **Low Perplexity (PPL):** The token is easily predicted from preceding words (e.g., "is", "the", boilerplate phrases). It carries little unique information -> **Safe to discard**.
- **High Perplexity (PPL):** The token is unexpected and contains critical, domain-specific information -> **Must preserve**.

---

### 2.2 The Complete LLMLingua Process

LLMLingua operates via three sequential components:

```
Original Prompt: (Instruction + Demonstrations + Question)
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: Budget Controller (Coarse-Grained Level)        │
│  - Protects Instruction (tau_ins = 0.85) & Question     │
│  - Calculates budget for Demonstrations (tau_dems)      │
│  - Ranks demonstrations by PPL & drops lowest           │
│  - Redistributes leftover budget to Instruction/Question│
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Iterative Token-Level Compression (ITPC)        │
│  - Splits prompt into segments (~100 tokens)            │
│  - Evaluates each token conditioned on previous kept    │
│    tokens using KV cache                                │
│  - Dynamically calculates threshold gamma_j per segment │
│  - Retains tokens where PPL > gamma_j                   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Distribution Alignment (Offline)                │
│  - Fine-tunes Small LM on target LLM instruction data   │
│    (Stanford Alpaca) so both models agree on "surprise" │
└─────────────────────────────────────────────────────────┘
```

#### Step 1: Budget Controller (Coarse-Grained Level)
A prompt is naturally split into three distinct components:
$$x = (x^{\text{ins}}, x^{\text{dems}}, x^{\text{que}})$$
- **Instruction (x_ins)** and **Question (x_que)** dictate the task and cannot tolerate aggressive cuts.
- **Demonstrations (x_dems)** provide pattern examples and contain substantial redundancy.

1. **Calculate Demonstration Budget:**  
   The controller fixes conservative retention rates for instruction and question:
   $$\tau_{\text{ins}} = 0.85, \quad \tau_{\text{que}} = 0.90$$
   The target retention rate for demonstrations ($\tau_{\text{dems}}$) is derived from the overall target rate $\tau$:
   $$\tau_{\text{dems}} = \frac{\tau \cdot L - (\tau_{\text{ins}} \cdot L_{\text{ins}} + \tau_{\text{que}} \cdot L_{\text{que}})}{L_{\text{dems}}}$$
2. **Demonstration Filtering:**  
   The small LM computes the average perplexity of each demonstration. Demonstrations are sorted from highest perplexity (most informative) to lowest perplexity. Demonstrations are selected until reaching a token cap:
   $$\text{Demonstration Token Limit} = k \cdot \tau_{\text{dems}} \cdot L_{\text{dems}} \quad (\text{with } k = 2)$$
3. **Budget Reallocation:**  
   Any leftover tokens from coarse demonstration filtering are transferred back to the instruction and question:
   $$\Delta\tau = \frac{k \cdot \tau_{\text{dems}} \cdot L_{\text{dems}} - \tilde{L}_D}{L_{\text{ins}} + L_{\text{que}}}$$

#### Step 2: Iterative Token-Level Compression (ITPC)
**The Problem with Naive Perplexity:**  
Computing perplexity for all tokens in a single forward pass assumes tokens are independent. However, language tokens are highly interdependent: deleting word A immediately alters whether word B makes sense.

**The ITPC Solution:**
1. Divide the prompt into sequential chunks of $S = 100$ tokens:
   $$S = \{s_1, s_2, \dots, s_m\}$$
2. For each chunk $s_j$, compute the perplexity of each token **conditioned on the compressed tokens already retained from previous chunks**:
   $$p(s_{j,i} \mid s_{j,<i}, \tilde{s}_{<j})$$
   *(The small LM uses its Key-Value (KV) cache so prior segments don't ne
   ed recomputing).*
3. Determine a dynamic cut-off threshold $\gamma_j$ corresponding to the target rate for that segment ($\tau_{\text{ins}} + \Delta\tau$, $\tau_{\text{dems}}$, or $\tau_{\text{que}} + \Delta\tau$).
4. Keep only tokens with perplexity above $\gamma_j$:
   $$\tilde{s}_j = \{ s_{j,i} \mid \text{PPL}(s_{j,i}) > \gamma_j \}$$

#### Step 3: Distribution Alignment
Small models (LLaMA-7B or GPT-2) and commercial target LLMs (GPT-3.5 or Claude) do not evaluate probabilities identically. To align them:
- Instruction-tune the small LM on 52k Alpaca examples generated by the target LLM.
- This minimizes the divergence between the small model's token predictions and the target LLM's expectations.

---

### 2.3 Key Findings & Limitations of LLMLingua
- **Strong Reasoning Retention:** On GSM8K math reasoning, achieves **5x compression with zero performance drop** (79.08 Exact Match vs 78.85 full-shot). Retains solid performance up to **20x compression** (77.33 EM).
- **Ablation Impact:**
  - Removing ITPC drops performance from 79.08 to 72.93 (-6.15 EM).
  - Removing Budget Controller drops performance to 73.62 (-5.46 EM).
  - Removing Distribution Alignment causes a slight drop to 78.62 (-0.46 EM).
- **Limitations:**
  1. Performance collapses steeply beyond 20x - 25x.
  2. Autoregressive causal models only see tokens from left to right, missing dependencies on subsequent words.
  3. Running a 7B causal model with iterative KV-cache management requires ~16.6 GB of VRAM and noticeable latency overhead.

---

## 3. Paper 2: LLMLingua-2 (Findings of ACL 2024)

### 3.1 The Paradigm Shift
LLMLingua-2 re-evaluates the prompt compression problem:
1. **Perplexity is a heuristic proxy**, not a metric trained for compression.
2. **Causal LMs are unidirectional**: They only see left context. In reality, whether a word should be kept often depends on what follows it.
3. **Solution:** Reframe prompt compression as a **supervised binary token classification task** (keep vs. discard) powered by a small **bidirectional Transformer encoder** (e.g., XLM-RoBERTa-large 355M or mBERT 110M).

---

### 3.2 The Complete LLMLingua-2 Pipeline

```
MeetingBank Transcripts (5,169 transcripts)
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: Data Distillation from GPT-4                    │
│  - Extractive prompt with 5 strict rules                │
│  - Chunk-wise compression (chunks <= 512 tokens)        │
│  - Dynamic compression (no artificial target ratio)     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Data Annotation (Algorithm 1)                   │
│  - Sliding window centered on previous match index      │
│  - Bidirectional search (checks left and right)         │
│  - spaCy lemmatization fuzzy matching (tenses/plurals)  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Quality Control & Filtering                     │
│  - Variation Rate (VR): Purge hallucinations (top 5%)   │
│  - Alignment Gap (AG = HR - MR): Purge bad mappings     │
│    (top 10%)                                            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: Model Training                                  │
│  - Train XLM-RoBERTa / mBERT with Cross-Entropy         │
│  - Predicts p(preserve) vs p(discard) for every token   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 5: Fast Inference Compression                      │
│  - Calculate p(preserve) in a single bidirectional pass │
│  - Retain top N_tilde = tau * N words                   │
│  - Reconstruct prompt preserving original word order    │
└─────────────────────────────────────────────────────────┘
```

#### Step 1: Data Distillation from GPT-4
To train a model, ground-truth (keep/discard) labels are needed. Asking human annotators is too expensive, while raw GPT-4 tends to paraphrase and hallucinate. Three constraints make GPT-4 a reliable distillation source:
1. **Strict 5-Rule System Prompt:**
   - Rule 1: Only remove unimportant words.
   - Rule 2: Do not reorder words.
   - Rule 3: Do not change words.
   - Rule 4: Do not use abbreviations or emojis.
   - Rule 5: Do not add any new words or symbols.
2. **Chunk-Wise Processing:** Long contexts (>512 tokens) cause GPT-4 to compress too aggressively, dropping key facts. All documents are chunked into <= 512 tokens ending at a sentence boundary.
3. **No Forced Compression Ratio:** Prompts vary in density. GPT-4 is prompted to compress "as short as possible while retaining maximum information," producing natural variations across sentences.

#### Step 2: Data Annotation (Algorithm 1)
GPT-4 outputs a string of text, not binary tags. Aligning the compressed text back to the original text involves three challenges:
- **Ambiguity:** Common words appear dozens of times.
- **Variations:** GPT-4 occasionally changes verb tense or plural forms.
- **Reordering:** Minor syntax shifts.

**Algorithm 1 Mechanics:**
- Uses a **sliding window** of size $s$ around the last confirmed match (`prev`).
- Performs a **bidirectional search** (checks both `prev + i` and `prev - i`).
- Uses **spaCy lemmatization** to match root lemmas (e.g., "rearranged" -> "rearrange").
- Sets the matched word's label to `True` (preserve).

#### Step 3: Quality Control Metrics
To eliminate noisy labels before training:
1. **Variation Rate (VR):** The proportion of words in the compressed text that do not exist in the original text (hallucination detector):
   $$\text{VR} = \frac{1}{|S_{\text{comp}}|} \sum_{w \in S_{\text{comp}}} \mathbb{I}(w \notin S_{\text{ori}})$$
   *Action:* Discard the top 5% worst samples.
2. **Alignment Gap (AG):** Evaluates how cleanly words mapped between texts:
   $$\text{Matching Rate (MR)} = \frac{1}{|S_{\text{ori}}|} \sum_{w \in S_{\text{ori}}} \mathbb{I}(\text{label}(w) == \text{True})$$
   $$\text{Hitting Rate (HR)} = \frac{1}{|S_{\text{ori}}|} \sum_{w \in S_{\text{comp}}} \mathbb{I}(w \in S_{\text{ori}})$$
   $$\text{AG} = \text{HR} - \text{MR}$$
   *Action:* Discard the top 10% samples with the widest gap.

#### Step 4: Compressor Architecture & Training
- **Backbone:** Bidirectional Transformer encoder $f_\theta$ (`xlm-roberta-large` 355M or `multilingual-bert` 110M) with a linear classification head:
  $$h = f_\theta(x), \quad p(x_i) = \text{Softmax}(W \cdot h_i + b)$$
- **Loss:** Binary cross-entropy over all words.
- **Subword Handling:** Subword probabilities are averaged to produce a single robust score per word.

#### Step 5: Test-Time Inference
For any user-specified compression rate $\tau$:
1. Calculate $p_i = p(\text{preserve})$ for each word in a single forward pass.
2. Retain the top $\tilde{N} = \tau \cdot N$ words with the highest probability.
3. Output the preserved words in their **original sequence order**.
- **Faithfulness:** Zero hallucination is guaranteed because words are only selected, never generated.

---

### 3.3 Key Findings & Performance Highlights
- **Speed & Efficiency:**
  - 3x - 6x faster compression latency than LLMLingua (0.4s vs 2.9s).
  - End-to-end latency reduction of 1.6x - 2.9x.
  - Peak GPU memory is **2.1 GB** (compared to 16.6 GB for LLMLingua and 26.5 GB for Selective-Context).
- **Benchmark Performance:**
  - **MeetingBank QA (3.1x ratio):** 86.92 EM (nearly matches original uncompressed 87.75 EM; outperforms LLMLingua's 67.52 EM).
  - **LongBench (5x ratio):** 39.1 average score (outperforms LLMLingua's 34.6).
  - **Reasoning (GSM8K, 5x ratio):** 79.08 EM (matches LLMLingua exactly).
- **Out-of-Domain & Multilingual Generalization:**
  - Trained solely on English meeting transcripts, but generalizes effectively to code, math reasoning, and Chinese benchmarks (LongBench-Zh: 38.1 vs LLMLingua's 28.6).

---

## 4. Head-to-Head Comparison

| Feature / Dimension | LLMLingua (EMNLP 2023) | LLMLingua-2 (Findings of ACL 2024) |
| :--- | :--- | :--- |
| **Core Philosophy** | Information entropy (unsupervised perplexity) | Supervised token classification ($p(\text{preserve})$) |
| **Context Processing** | Causal LM (Unidirectional, Left-only) | Transformer Encoder (Full Bidirectional) |
| **Compressor Architecture** | LLaMA-7B / Alpaca-7B / GPT-2 | XLM-RoBERTa-large (355M) / mBERT (110M) |
| **GPU Memory Footprint** | High (~16.6 GB - 26.5 GB) | Lightweight (~2.1 GB, 8x lower) |
| **Inference Mechanism** | Iterative segment re-scoring (ITPC) + KV cache | Single forward pass + Top-K word selection |
| **Compression Latency** | ~2.1s - 2.9s | ~0.4s - 0.5s (3x - 6x faster) |
| **End-to-End Speedup** | 1.7x - 5.7x | 1.6x - 2.9x |
| **Faithfulness Guarantee** | Extractive (Safe) | Extractive (Safe) |
| **Optimal Compression Range** | High ratios (5x - 20x) for few-shot ICL | Moderate ratios (2x - 5x) for general context |
| **Primary Limitation** | Unidirectional bias, heavy model, slow inference | Task-agnostic (does not condition on question) |

---

## 5. Lab Project Architecture & Roadmap

### The Central Research Question
> **"Can we compress prompts adaptively across different structural segments rather than applying a single uniform compression ratio?"**

In real-world applications, a prompt consists of heterogeneous parts:
```
[Instruction]   --> Very high sensitivity (losing instructions causes catastrophic failure)
[Demonstrations]--> Moderate sensitivity (examples can be summarized or trimmed)
[Retrieved Doc] --> Variable sensitivity (some passages are relevant, others are noise)
[Question]      --> Extreme sensitivity (must preserve entities and intent)
```

Uniform compression applies the same cut rate $\tau$ to everything. Your project explores **adaptive segment budgeting** to optimize task accuracy per token spent.

---

### Implementation Levels & Milestones

```
Level 1: Tooling & Verification (Week 1)
 ├── Set up Python environment with PyTorch and Hugging Face transformers
 ├── Install `llmlingua` library
 └── Run standalone compression test using pre-trained LLMLingua-2

Level 2: Baseline Pipeline & Benchmarks (Week 2)
 ├── Build Baseline 1: Naive Truncation (Head/Tail token pruning)
 ├── Build Baseline 2: Sentence Selection (BM25 / Sentence-Transformers similarity)
 ├── Integrate LLMLingua-2 (`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`)
 └── Test on GSM8K (Reasoning) and HotpotQA/MeetingBank (QA)

Level 3: Cost-Performance Evaluation (Week 3)
 ├── Sweep compression ratios: 1x (Original), 2x, 3x, 5x, 10x
 ├── Measure downstream accuracy (Exact Match / F1) using target LLM
 ├── Track real token savings, compression latency, and API cost
 └── Plot Performance vs. Compression Ratio curves

Level 4: Adaptive Segment-Wise Compression (Week 4 - Core Research)
 ├── Segment the prompt automatically: Instruction, Demonstrations/Docs, Question
 ├── Assign differentiated budgets (e.g., tau_ins=0.9, tau_que=1.0, tau_context=0.2)
 └── Compare Adaptive Compression vs. Uniform Compression under identical total token budgets

Level 5 & 6: Bonus / Stretch Goals (Week 5)
 └── Automatic segment sensitivity estimation or query-conditioned re-weighting
```

---

## 6. Practical Next Steps to Start Coding

1. **Create Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # or cpu
   pip install transformers datasets accelerate sentencepiece llmlingua
   ```

2. **Execute First Reproduction Test:**
   Run a short script loading `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` to compress a sample text at $\tau = 0.33$ and inspect the preserved tokens.

3. **Build the Baseline Module (`baselines.py`):**
   Implement `truncate_prompt(text, target_tokens)` and `sentence_selection_bm25(text, query, target_tokens)`.
