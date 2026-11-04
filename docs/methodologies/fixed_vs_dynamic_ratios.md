# Fixed Ratios vs. Dynamic Ratios: How Budgets Are Allocated

**User Question:** *"In these approaches, are we using fixed ratios?"*

---

## 1. Quick Summary Matrix

| System | Overall User Target | Ratio Per Segment / Document | Is it Fixed or Dynamic? |
| :--- | :---: | :---: | :---: |
| **Our Benchmark so far (`benchmark_hotpotqa.py`)** | $\tau = 0.50$ (or $0.33$) | Every single document gets the **exact same** $\tau = 0.50$ | **STRICTLY FIXED (Uniform)** |
| **LLMLingua-2 (ACL 2024)** | User-defined $\tau$ | Applied **uniformly across all words** indiscriminately | **STRICTLY FIXED (Uniform)** |
| **LLMLingua-1 (EMNLP 2023)** | User-defined $\tau$ | Hardcoded rules: $\tau_{\text{ins}}=0.85$, $\tau_{\text{que}}=0.90$, residual to demos | **SEMI-FIXED (Rule-Based)** |
| **LongLLMLingua (ICLR 2024)** | User-defined $\tau$ | Calculated per document: $\tau_i = f(\text{relevance}_i)$ | **TRULY DYNAMIC (Non-Uniform)** |
| **Our Proposed Adaptive Method** | User-defined $\tau$ | Gold docs get $\tau \approx 0.85$, distractors get $\tau \le 0.10$ | **TRULY DYNAMIC (Semantic RAG)** |

---

## 2. Detailed Breakdown of Each Approach

### A. What We Did in Our Benchmark So Far (Strictly Fixed)
In `benchmark_hotpotqa.py`, when we ran `rate = 0.50`:
- Document 1 (Gold Evidence): compressed at $\tau = 0.50$
- Document 2 (Distractor Noise): compressed at $\tau = 0.50$
- Document 3 (Distractor Noise): compressed at $\tau = 0.50$
- ...
- Document 10 (Gold Evidence): compressed at $\tau = 0.50$

**Why it failed:**  
Because the ratio was **fixed and blind**, Document 2 (useless noise) was allocated the exact same token budget as Document 1 (the actual answer). Half the remaining prompt was wasted on noise.

---

### B. LLMLingua-1: Semi-Fixed (Coarse Rules)
LLMLingua-1 does **not** compute a flexible ratio per passage. Instead, it uses **hardcoded numbers**:
1. It hardcodes $\tau = 0.85$ for instructions.
2. It hardcodes $\tau = 0.90$ for questions.
3. Whatever token budget is left over is divided among the few-shot demonstrations:
   - It ranks demonstrations by perplexity.
   - It keeps the top demonstrations and completely deletes the rest.

*Verdict:* It is **semi-fixed**. It uses rigid, hand-crafted rules that fail in RAG scenarios.

---

### C. LongLLMLingua: Truly Dynamic Ratios
In LongLLMLingua (ICLR 2024), ratios are **not fixed**. Each document receives its own custom compression rate $\tau_i$ based on how relevant it is to the question:

Suppose the user asks for an overall target compression of **$3\times$ ($\tau_{\text{total}} = 0.33$)** on 10 documents:

```
Document 1 (Gold Evidence):        Relevance = 0.95  ──► Assigned tau_1 = 0.85 (Gentle pruning)
Document 2 (Irrelevant Arkansas):  Relevance = 0.12  ──► Assigned tau_2 = 0.00 (Hard drop)
Document 3 (Irrelevant Film):      Relevance = 0.18  ──► Assigned tau_3 = 0.05 (Aggressive cut)
...
Document 10 (Gold Evidence):       Relevance = 0.91  ──► Assigned tau_10 = 0.80 (Gentle pruning)
```

**The Math of Dynamic Budgeting:**
The individual document rates $\tau_i$ are completely different, but their **weighted sum still meets the user's overall target**:
$$\sum_{i=1}^{10} \tilde{L}_i = \tau_{\text{total}} \times L_{\text{total}}$$

*Benefit:* The user still gets a $3\times$ smaller prompt, but **100% of the token budget is concentrated on the passages that actually contain the answer**.

---

## 3. The Core Innovation for Our Lab Project

The goal of our lab project is to build this **Dynamic Non-Uniform Allocation** into LLMLingua-2:

```
[Target: Overall tau = 0.33 (3x compression)]
                     │
                     ▼
       ┌───────────────────────────┐
       │ Fast Semantic Scorer      │
       │ (Cosine Similarity in ms) │
       └─────────────┬─────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  [Gold Passages]        [Distractor Noise]
Relevance: HIGH (0.8+)  Relevance: LOW (<0.3)
         │                       │
         ▼                       ▼
    tau_gold = 0.85        tau_distractor = 0.05
 (Keep 85% of facts)    (Drop 95% of the noise)
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
[Overall Prompt Meets Target tau = 0.33, with 90%+ Evidence Preserved!]
```
