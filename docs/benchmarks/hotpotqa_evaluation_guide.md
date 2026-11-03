# HotpotQA Multi-Document RAG: Evaluation & Metrics Guide

This guide explains how prompt compression is evaluated in multi-document RAG tasks, breaking down the question types, evidence metrics, and how to interpret experimental results.

---

## 1. Core Evaluation Concepts

### 1.1 Facts (Premises) vs. Answers (Deductions)
A prompt compressor is a **context condenser**, not an answer generator. Its goal is to provide a shortened context to a downstream reader LLM (like GPT-4 or Llama-3).

* **Gold Standard:** Human-verified ground truth.
* **Gold Evidence (Supporting Facts):** The exact sentences in a dataset labeled by human annotators as containing the proof required to answer the question.
* **Distractors:** Irrelevant documents retrieved alongside the gold documents to test the system's ability to filter noise.

---

## 2. HotpotQA Question Types: Bridge vs. Comparison

HotpotQA divides its benchmark questions into two distinct categories:

| Question Type | % of Dataset | How Answer Works | Example | Evaluation Rule |
| :--- | :---: | :--- | :--- | :--- |
| **Bridge Questions** | **~80%** | The answer is an **exact text span** (named entity, date, number) copied verbatim from the document. | Question: *"What company formed the boy band?"*<br>Answer: **`"YG Entertainment"`** | Evaluated via **Extractive Answer Recall**: The exact entity words must survive compression. |
| **Comparison Questions** | **~20%** | The answer is an inferential deduction (**"yes" / "no"**). The word "yes" does not exist in the text. | Question: *"Were Derrickson and Ed Wood of the same nationality?"*<br>Answer: **`"yes"`** | Evaluated via **Gold Evidence Retention**: Both premises (*American* + *American*) must survive. |

---

## 3. The Two Key Diagnostic Metrics

Before spending money on expensive downstream LLM API calls, we evaluate the compressor using two metrics:

### A. Gold Evidence Retention Rate
$$\text{Gold Retention} = \frac{\text{Number of Gold Words Kept}}{\text{Total Gold Words in Original Evidence}}$$
* If this drops below 50%, no downstream LLM can answer the question because the core evidence was deleted.

### B. Distractor Retention Rate & Selectivity Gap
$$\text{Distractor Retention} = \frac{\text{Number of Distractor Words Kept}}{\text{Total Distractor Words in Context}}$$
$$\text{Selectivity Gap} = \text{Gold Retention} - \text{Distractor Retention}$$
* Measures whether the compressor actively filters out noise. A positive gap means the model prioritizes real evidence over distractors.

---

## 4. The "Entity Fragility" Problem & Why Type-Gating is Required

Consider this evidence sentence:
> *"The arena where the Lewiston Maineiacs played can seat **3,677 seated** people."*

If the compressor trims it to:
> *"arena where Lewiston Maineiacs played can seat people"*

- **Evidence Retention:** 80% (most words survived).
- **Extractive Answer Retention:** 0% (the specific number `"3,677 seated"` was deleted!).

Without checking entity survival on extractive questions, benchmarks falsely claim high quality while the single fact the user asked for was destroyed. 

In our benchmark script (`benchmark_hotpotqa.py`), we use **Type-Gated Checking**:
- **Bridge questions:** Verifies that fragile named entities survived.
- **Comparison questions:** Evaluates premise survival, avoiding false negatives on boolean answers.

---

## 5. Decoding Common Terminal Warnings

| Warning Message | Plain English Meaning | Severity |
| :--- | :--- | :---: |
| `FutureWarning: Using TRANSFORMERS_CACHE is deprecated...` | Hugging Face updated their library settings and prefers `HF_HOME`. | Harmless developer note. |
| `UserWarning: huggingface_hub cache-system uses symlinks...` | Windows restricts symlinks without Developer Mode; Hugging Face copies files normally on Drive E:. | Harmless file copy fallback. |
| `Token indices sequence length is longer than specified maximum (1739 > 512)...` | Long multi-document prompt exceeded BERT's 512 window; LLMLingua-2 automatically split it into chunks. | Informational; handled automatically. |
