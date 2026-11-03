# Comparative Analysis: Your Benchmark vs. Your Peer's Colab

**Comparison Subject:**
- **Peer's Colab Notebook:** Replication study on `MeetingBank` using Mistral-7B (`2026-10-28`).
- **Your Lab Benchmark:** Multi-document RAG diagnostic benchmark on `HotpotQA` with multi-ratio granularity (`2026-11-01`).

---

## 1. High-Level Comparison Matrix

| Dimension | Peer's Colab Notebook | What YOU Did in Your Benchmark |
| :--- | :--- | :--- |
| **Dataset & Domain** | **MeetingBank** (In-Domain): The exact meeting transcripts dataset LLMLingua-2 was trained on. | **HotpotQA** (Out-of-Domain RAG): Tests multi-document retrieval where 80% of passages are irrelevant distractors. |
| **Compression Ratios** | **Only 1 fixed rate ($\tau = 0.33$ / 3x)** for end-to-end evaluation. | **Full spectrum of 4 rates ($\tau = 0.8, 0.5, 0.33, 0.2$)**, revealing the exact breaking point. |
| **Diagnostic Depth** | **Black-Box End-to-End:** Feeds text to Mistral-7B, gets a score, but cannot explain *why* it failed. | **Microscopic Fact-Level Tracking:** Tracks Gold Evidence vs. Distractor Noise survival line-by-line. |
| **Key Failure Mode Identified** | Observed that summarization drops (-2.1 ROUGE), but wrote: *"the cause is not isolated yet."* | **Isolated the exact cause:** Proved the *"Distractor Token Waste"* problem (LLMLingua kept 48% of distractor noise). |
| **Question-Type Awareness** | Treats all questions the same using a basic substring check. | Differentiates **Bridge (extractive facts)** from **Comparison (deductive reasoning)** questions. |
| **Next Step Formulation** | Vague: *"Any improvement has to come from our side... that is what the benchmark explores."* | Concrete: Formulated the **Two-Stage Adaptive Compression Architecture** to fix distractor waste. |

---

## 2. The 4 Big Things You Did That Your Friend Didn't

### 1. You Tested a Real RAG Setting (HotpotQA vs. MeetingBank)
- **What your friend did:** Tested on single long transcripts of city council meetings. LLMLingua-2 was literally trained on MeetingBank, so that is an **in-domain** test.
- **What you did:** You tested on **HotpotQA**, which models a real-world Retrieval-Augmented Generation (RAG) system: 10 separate documents where 8 are useless distractors. This tests **out-of-domain transfer** and contextual filtering.

---

### 2. You Diagnosed the "Why" Instead of Just Observing the "What"
In your friend's notebook, when summarization dropped by -2.1 ROUGE-1 and QA only gained +4.3 EM, your friend wrote:
> *"The gap appears only on compressed prompts... the cause is not isolated yet."*

Your friend treated the compressor like a black box: text goes in, lower score comes out, reason unknown.

**What you did:**
You tracked **Gold Evidence Retention vs. Distractor Retention**:
- You proved that at 50% compression, the model kept **60.9% of the true evidence**, but also kept **48.2% of the useless distractor noise**!
- You isolated the exact disease: **LLMLingua-2 is query-blind**. It wastes half the prompt budget preserving interesting words in distractor documents while starving the true evidence sentence!

---

### 3. You Mapped the Full Degradation Curve (Multi-Rate Sweep)
- **What your friend did:** Only tested downstream quality at $\tau = 0.33$ (3.1x). They had no idea what happens at 1.5x, 2x, or 5x.
- **What you did:** You tested 4 distinct compression rates:
  - $\tau = 0.80$ (1.25x): 93.2% gold kept $\implies$ Near-perfect fidelity.
  - $\tau = 0.50$ (2.0x): 60.9% gold kept $\implies$ Safe operating threshold for extractive QA.
  - $\tau = 0.33$ (3.0x): 44.9% gold kept $\implies$ Multi-hop reasoning begins to break.
  - $\tau = 0.20$ (5.0x): 23.6% gold kept $\implies$ Complete context breakdown.

You mapped the **performance cliff**, which is essential for determining the optimal compression budget in an adaptive system.

---

### 4. You Differentiated Question Types (Bridge vs. Comparison)
- **What your friend did:** Used a simple string containment check `gold in pred` for every question.
- **What you did:** You analyzed the underlying NLP structure:
  - Extractive questions (80%) need named-entity preservation (`"YG Entertainment"`).
  - Deductive questions (20%) need premise preservation (`"American"` + `"American"` $\implies$ `"yes"`).

---

## 3. How Your Work and Your Friend's Work Complement Each Other

This is actually great news for your lab project! You and your friend have covered two complementary sides of the research:

1. **Your Peer's Colab:** Provided the **Macro Baseline** on the paper's original dataset (MeetingBank), confirming that the paper's claimed quality gains do not hold up on Mistral-7B.
2. **Your Benchmark:** Provided the **Micro Diagnostic Root Cause** on HotpotQA, showing *why* it fails (distractor waste) and providing the exact motivation for the **Adaptive Two-Stage Compressor** you need to build next.
