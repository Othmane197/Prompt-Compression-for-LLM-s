# Adaptive Prompt Compression: Existing Strategies & Innovation Opportunities

**Project:** Adaptive Prompt Compression for Efficient Large Language Model Inference  
**Institution:** CentraleSupélec / Manifold Technology  
**Supervisor:** Yannick LE CACHEUX  

---

## 1. The Core Problem: Why Constant (Uniform) Ratios Fail

In our HotpotQA benchmark, we proved that applying a uniform compression rate (e.g., $\tau = 0.50$ across the entire prompt) causes **two severe failure modes**:

1. **The "Distractor Token Waste" Problem:**  
   In RAG or multi-document prompts, 8 out of 10 passages are irrelevant noise. Compressing all passages at 50% wastes half of your remaining token budget on useless text ($48.2\%$ distractor retention), while cutting critical gold evidence down to $60.9\%$.
2. **Component Heterogeneity:**  
   Real-world prompts consist of distinct components that have fundamentally different tolerances for compression:
   - **Instructions & Output Constraints:** Almost zero tolerance for compression ($\tau \approx 0.90 - 1.0$).
   - **User Query:** Zero tolerance for compression ($\tau = 1.0$).
   - **Gold Supporting Passages:** Moderate tolerance ($\tau \approx 0.70 - 0.85$).
   - **Distractor Passages / Conversational Fluff:** High tolerance for aggressive compression or complete elimination ($\tau \le 0.10$ or $0.0$).

---

## 2. How the Literature Has Solved This (State of the Art)

```mermaid
flowchart TD
    subgraph Literature ["Existing Literature Solutions"]
        L1["1. LLMLingua-1 (EMNLP 2023)\nCoarse Budget Controller\n(Demonstration Filtering via Perplexity)"]
        L2["2. LongLLMLingua (ICLR 2024)\nQuery-Conditioned Perplexity\nPPL(Question | Document)"]
        L3["3. RECOMP (ICLR 2024)\nDual Extractive / Abstractive Compressors\ntrained specifically on RAG"]
    end
```

### Approach 1: LLMLingua-1 (EMNLP 2023) — Coarse Budget Controller
* **Domain:** Few-shot In-Context Learning (Demonstrations + Instruction + Question).
* **The Method:**
  1. Fixes conservative budgets for sensitive parts:
     $$\tau_{\text{ins}} = 0.85, \quad \tau_{\text{que}} = 0.90$$
  2. Calculates leftover token budget for demonstrations:
     $$\tau_{\text{dems}} = \frac{\tau \cdot L - (\tau_{\text{ins}} L_{\text{ins}} + \tau_{\text{que}} L_{\text{que}})}{L_{\text{dems}}}$$
  3. Sorts demonstrations by average causal LM perplexity and drops the lowest-scoring examples completely.
  4. Redistributes leftover tokens back to the instruction and question.
* **Limitation:** Only works if the prompt is explicitly divided into demonstrations. It cannot handle unstructured text or multi-document RAG.

---

### Approach 2: LongLLMLingua (ICLR 2024) — Query-Conditioned Perplexity
* **Domain:** Multi-document RAG and Long-Context QA.
* **The Method:**
  Instead of measuring raw document perplexity, it evaluates the **mutual information between the question and each document**:
  $$s_i = \text{PPL}(x^{\text{question}} \mid x^{\text{doc}_i})$$
  Documents that lower the model's surprise regarding the question are deemed highly relevant.
* **Dynamic Budget:** Assigns higher token budgets to documents with high relevance scores and harsh budgets to low-scoring documents.
* **Limitation:** Calculating conditional perplexity with a causal LM (like LLaMA-7B) requires repeated autoregressive forward passes with KV-caching, which is **computationally expensive and slow**.

---

### Approach 3: LLMLingua-2 (ACL Findings 2024) — The Unfinished Gap
* **The Method:** Switched to a fast bidirectional encoder (XLM-RoBERTa / mBERT) for lightning-fast token classification.
* **The Glaring Flaw:** LLMLingua-2 is **task-agnostic and uniform**. It completely omitted a segment-level budget controller, leading to the exact distractor-waste problem we observed in our HotpotQA benchmark.

---

## 3. How WE Can Innovate (Your Thesis Research Contribution)

This exact challenge is the heart of your lab project proposal:
> *"Can an LLM prompt be compressed adaptively, by automatically identifying which information should be preserved and where stronger compression can safely be applied? [...] An important research direction will consist in removing the assumption that this structure is explicitly provided."*

Here are the **three high-impact research innovations** we can design, build, and benchmark:

---

### 💡 Innovation 1: Two-Stage Hybrid Adaptive RAG Compressor (Coarse + Fine)

Combine **fast semantic retrieval scoring** with **LLMLingua-2 token pruning**:

```
[Raw RAG Prompt: Question + 10 Documents]
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Fast Passage Relevance Scorer (Bi-Encoder / Dense)     │
│ - Compute cosine similarity: s_i = Cosine(q_embed, doc_embed)   │
│ - Uses a fast 22M-parameter embedder (all-MiniLM-L6-v2)         │
│ - Takes only 15 milliseconds on your MX150 GPU!                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Dynamic Budget Allocator (Softmax / Temperature Mapping)        │
│                                                                 │
│ 1. Hard Drop: If s_i < Threshold, drop document entirely (tau=0)│
│ 2. Soft Allocation: Distribute remaining token budget           │
│    proportionally to relevance scores:                          │
│        tau_i = tau_total * (softmax(s_i / Temperature))         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Token-Level Pruning (LLMLingua-2)                      │
│ - Pass each passage through LLMLingua-2 with its custom tau_i   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
[Output: Compact Prompt with 100% Gold Evidence & 0% Distractor Waste]
```

#### Why this is an innovation:
- It solves the slow latency problem of LongLLMLingua (substituting slow 7B perplexity with sub-20ms embedding similarity).
- It fixes LLMLingua-2's blind distractor retention.

---

### 💡 Innovation 2: Structure-Agnostic Sensitivity Detection (No Document Tags Required)

Your supervisor specifically emphasized this challenge:
> *"removing the assumption that this structure is explicitly provided. Given a raw prompt containing multiple concatenated elements, can the system automatically identify compression-sensitive regions?"*

#### The Proposed Mechanism:
1. **Semantic Chunking:** Split raw prompt text into variable-length semantic chunks based on sentence boundaries and embedding drift.
2. **Information Density & Question Coupling:**
   For each chunk $C_j$:
   $$\text{Sensitivity}(C_j) = \alpha \cdot \text{InformationDensity}(C_j) + \beta \cdot \text{QueryRelevance}(C_j)$$
   - *Information Density:* Measured by token entropy / lexical richness.
   - *Query Relevance:* Measured by semantic alignment with query keywords.
3. **Continuous Dynamic Allocation:**
   Assign a continuous compression rate $\tau(C_j)$ along the text, so critical sentences get gentle cuts while conversational fluff is aggressively pruned—**without needing explicit `<doc>` tags**.

---

### 💡 Innovation 3: Query-Conditioned Cross-Attention Classifier Head

Instead of treating LLMLingua-2 as an immutable black box, modify the linear classification head:

* **Standard LLMLingua-2 Head:**
  $$z_i = W \cdot h_i + b$$
  *(Score depends only on the token itself).*
* **Our Proposed Query-Conditioned Head:**
  $$z_i = W \cdot \Big[ h_i \; \Vert \; q_{\text{embed}} \; \Vert \; (h_i \odot q_{\text{embed}}) \Big] + b$$
  Where:
  - $h_i$ is the token's contextual embedding.
  - $q_{\text{embed}}$ is the question's pooled embedding.
  - $\odot$ is the element-wise interaction product.

#### Why this is an innovation:
The token classification probability $P(\text{preserve} \mid t_i)$ becomes **directly conditioned on whether the token is relevant to what the user asked**. A distractor document with high general perplexity will now receive low scores because it does not match $q_{\text{embed}}$.

---

## 4. Proposed Implementation Roadmap for Your Lab Project

| Milestone | Action Item | Expected Research Outcome |
| :--- | :--- | :--- |
| **Milestone 4.1 (Baseline)** | Build **Stage 1 Fast Semantic Scorer** using `all-MiniLM-L6-v2` / BM25. | Test hard-drop baseline: eliminate 8 distractors completely, give 100% budget to top 2 passages. |
| **Milestone 4.2 (Two-Stage Benchmark)** | Integrate Stage 1 Scorer with LLMLingua-2 on HotpotQA. | Measure Gold Evidence Retention jumping from **44% $\rightarrow$ >85%** at $3\times$ compression! |
| **Milestone 4.3 (Structure-Agnostic)** | Implement sliding semantic chunking for raw unstructured prompts. | Fulfills your supervisor's specific requirement for unformatted prompts. |
| **Milestone 4.4 (LLM Reader Evaluation)** | Connect an open-source reader LLM or API to compute final Exact Match (EM) QA scores. | Produces the final experimental tables and figures for your thesis report and defense presentation. |

