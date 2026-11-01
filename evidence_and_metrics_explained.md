# Understanding "Gold Evidence" and "Answer Retention" in Prompt Compression Research

**Topic:** Research Evaluation Methodology for Prompt Compression  
**Papers Referenced:**
- *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models* (EMNLP 2023)
- *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression* (ACL Findings 2024)
- *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering* (EMNLP 2018)

---

## 1. Where Does the Word "Gold" Come From?

In Artificial Intelligence and Natural Language Processing, the term **"Gold"** comes from the phrase **"Gold Standard"**:
- **Gold Standard = The 100% verified ground truth created by human experts.**
- **Gold Evidence (or Gold Supporting Facts):** The exact sentences in a dataset that humans manually labeled as containing the proof needed to answer the question.

### The HotpotQA Structure:
When Stanford and Carnegie Mellon researchers created HotpotQA, they gave each question **10 documents** (around 40 to 60 sentences total):
- **38 to 58 sentences are "Distractors":** Random, irrelevant Wikipedia articles designed to confuse the model.
- **2 to 4 sentences are "Gold Supporting Facts":** The only sentences that actually hold the answer.

> **Example:**  
> - **Question:** *"Were Scott Derrickson and Ed Wood of the same nationality?"*  
> - **Gold Sentence 1:** *"Scott Derrickson is an American director..."*  
> - **Gold Sentence 2:** *"Ed Wood was an American filmmaker..."*  
> - **Distractor Sentences (Noise):** 34 irrelevant sentences about a town called Woodson Arkansas, movie awards, box office numbers, etc.

---

## 2. Did the Authors Do This in Their Research Papers?

### Yes, but in two complementary ways:

### A. How the Papers Evaluated with a Target LLM (End-to-End QA)
In both LLMLingua-1 and LLMLingua-2, the authors passed the compressed text to a large target LLM (such as **GPT-3.5-Turbo**, **GPT-4**, or **Llama-2-70B**) and evaluated:
1. **Exact Match (EM):** Did the LLM produce the exact ground-truth answer string?
2. **F1 Score:** What percentage of words in the LLM's answer overlapped with the ground-truth answer?

### B. Why We Evaluated "Gold Evidence Retention" in Our Benchmark
Calling GPT-4 on thousands of compressed prompts costs hundreds of dollars in API fees and takes hours.

Before you spend API credits on GPT-4, you must first answer a fundamental diagnostic question:  
> **"Did the compressor actually preserve the facts necessary for GPT-4 to answer, or did it accidentally delete them?"**

This is why measuring **Gold Evidence Retention** is standard practice in prompt compression and RAG research:
1. **Gold Evidence Retention Rate:**  
   $$\text{Gold Retention} = \frac{\text{Number of Gold Words Kept in Compressed Prompt}}{\text{Total Number of Gold Words in Original Evidence}}$$
   If this drops below 50%, no LLM in the world can answer the question because the information was erased from the prompt!
2. **Distractor Retention Rate:**  
   $$\text{Distractor Retention} = \frac{\text{Number of Distractor Words Kept}}{\text{Total Number of Distractor Words}}$$
   If this is high, the compressor is wasting valuable token budget on garbage passages.
3. **Selectivity Gap ($\text{Gold} - \text{Distractor}$):**  
   Measures whether the compression model is "smart" (retaining more true evidence than noise) or "blind" (pruning both equally).

---

## 3. What Did "Ans Kept" (Answer Recall) Measure?

`Ans kept` is a strict sanity check:
- It checks if the literal words of the ground-truth answer (e.g. `"Chief of Protocol"` or `"YG Entertainment"`) survived the compression.
- **Why it matters:** If a prompt compressor prunes the word `"YG Entertainment"` out of the context passage, an LLM reading that prompt has no way of knowing who formed the boy band without hallucinating.

---

## 4. What Our Benchmark Proved

By measuring **Gold Evidence Retention vs. Distractor Retention**, our benchmark revealed the biggest scientific insight for your thesis:

| Metric at $\tau = 0.50$ (2x Compression) | What Happened | What It Proves |
| :--- | :--- | :--- |
| **Gold Evidence Retained** | **60.9%** | The compressor preserved over 60% of the true facts. |
| **Distractor Noise Retained** | **48.2%** | **The compressor still kept almost half of the useless distractor words!** |

### The Conclusion for Your Project:
LLMLingua-2 does not know what the user is asking. It treats a distractor sentence with "interesting" words with the same importance as the gold sentence containing the answer.

This directly justifies your lab project's objective: building an **Adaptive Compressor** that drops the distractor documents entirely and preserves 100% of the gold evidence.
