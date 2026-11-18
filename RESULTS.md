# Results

Measured results only. Every number below can be recomputed from the JSON files in [`results/`](results/).
Sample sizes are small: these are **pilot** results, not conclusions.

---

## 1. Stability pilot: reader performance vs compression (main result)

**Setup:** HotpotQA (distractor), validation queries 0–19 · compressor LLMLingua-2
(`xlm-roberta-large-meetingbank`, documents only; instruction and question kept intact) ·
reader Qwen2.5-3B-Instruct (float16, Kaggle T4) · per (query, rate): 1 greedy answer +
5 sampled answers (temperature 0.7) · scores: Exact Match / F1 (official HotpotQA normalization).
Script: `pilot_map_stability.py` · data: `results/pilot_3b.json`.

| Target rate | Actual ratio | Greedy EM | Greedy F1 | Sampled F1 (mean of 5) | F1 std across the 5 runs |
|---|---|---|---|---|---|
| 1.00 | 1.00× | 0.40 | 0.50 | 0.53 | 0.024 |
| 0.80 | 1.52× | 0.35 | 0.49 | 0.49 | 0.007 |
| 0.50 | 3.24× | 0.25 | 0.36 | 0.40 | 0.039 |
| 0.33 | 4.73× | 0.25 | 0.42 | 0.39 | 0.070 |

**Findings**

1. **Run-to-run variability is small.** In 70 of 80 (query, rate) cells, the 5 sampled answers
   get the same score. Average spread (0.007–0.070 F1) is smaller than the compression effect
   (~0.13 F1 at 3.2×). Greedy decoding is deterministic. → One greedy answer per (query, rate)
   is enough for this reader and setting.
2. **Per-query curves are not monotonic.** Compression sometimes *helps*: at 3.2×, 6 queries
   lose >0.2 F1 but 3 queries gain >0.2 F1. Examples (greedy F1 at 1× / 1.5× / 3.2× / 4.7×):
   query 9: 1 / 1 / 0 / 1; query 8: 0 / 0 / 0 / 1. → "Maximum ratio before losing 5%" is
   ill-defined for a single query; labels need smoothing or a different formulation.
3. **The average drop is not statistically established.** Mean F1 drop 1× → 3.24× = 0.13,
   95% bootstrap CI [−0.08, 0.33] (20 queries). Several hundred queries are needed.
4. **Weak baseline.** Only 11/20 queries reach F1 ≥ 0.5 without compression; queries already
   wrong at 1× carry no information about the tolerable ratio.

## 1b. Same pilot with a stronger reader: Qwen2.5-7B (full float16)

**Setup:** identical to §1 (same 20 queries, same rates, **identical compressed inputs** —
verified: same actual ratios), reader Qwen2.5-7B-Instruct in float16 split over 2 × T4.
Data: `results/pilot_7b.json` · comparison: `python plot_results.py results/pilot_3b.json --compare results/pilot_7b.json`.

| Target rate | Actual ratio | Greedy EM | Greedy F1 | Sampled F1 (mean of 5) | F1 std across runs |
|---|---|---|---|---|---|
| 1.00 | 1.00× | 0.50 | 0.59 | 0.59 | 0.004 |
| 0.80 | 1.52× | 0.20 | 0.42 | 0.42 | 0.006 |
| 0.50 | 3.24× | 0.35 | 0.48 | 0.47 | 0.022 |
| 0.33 | 4.73× | 0.20 | 0.33 | 0.37 | 0.052 |

**Findings**

1. **Non-monotonic curves persist with the stronger reader.** Queries whose F1 rises by >0.2
   between two consecutive ratios: 3B 7/20, 7B 6/20, **same query in both: 3/20** (Q2, Q6, Q16).
   Most jumps are therefore reader-specific; a few appear with both readers on identical inputs.
2. **The "max safe ratio" depends on the reader.** Same query, same compressed texts, different
   label: Q3 → 3B 3.40×, 7B 1.00×; Q9 → 3B 1.37×, 7B 4.91×. A ratio estimator is reader-specific.
3. **Label rules disagree slightly more with the 7B:** 2 of 12 usable queries (Q6, Q10) vs 0 of 11
   for the 3B (sampled scores).
4. **Run-to-run variability stays small** (std ≤ 0.052 F1; at most 1/20 queries flip).
5. Higher uncompressed baseline (F1 0.59 vs 0.50), but a sharper drop at 1.5× (0.42 vs 0.49).
   All on 20 queries: indicative only.

## 2. Reader choice (uncompressed baseline, 20 queries, greedy)

| Reader | EM | F1 | Runtime (Kaggle T4) |
|---|---|---|---|
| Qwen2.5-3B-Instruct (float16) | 0.40 | 0.50 | 132 s |
| Qwen2.5-7B-Instruct (4-bit) | 0.45 | 0.54 | 265 s |

A 1-query difference is 0.05, so the gap is not significant; the 3B was kept (2× faster).
A float32 vs float16 check on Qwen2.5-3B (5 queries × 4 rates) gave identical answers
(`results/diag_fp32.json`, `results/diag_fp16.json`).

## 3. Compression-only benchmark (no reader)

**Setup:** HotpotQA, 10 queries · LLMLingua-2 `bert-base-multilingual-cased-meetingbank` ·
lexical proxies only (no LLM answers). Script: `benchmark_hotpotqa.py` ·
data: `results/hotpotqa_benchmark_results.json`.

| Target rate | Actual ratio | Answer string kept (7 extractive questions) | Gold-evidence words kept | Distractor words kept |
|---|---|---|---|---|
| 0.80 | 1.67× | 100% | 93% | 80% |
| 0.50 | 3.55× | 86% (6/7) | 61% | 48% |
| 0.33 | 5.41× | 43% | 45% | 32% |
| 0.20 | 10.96× | 14% | 24% | 15% |

**Findings:** LLMLingua-2 overshoots the requested rate (2× requested → 3.55× obtained) and
barely separates useful from distractor text. Caveat: "words kept" counts a word as kept if it
appears anywhere in the compressed text, which overestimates both retention rates.

---

## Known limitations

- Small samples (20 and 10 queries); single dataset (HotpotQA); single compressor family.
- Official EM/F1 normalization penalizes format differences ("1969-1974" vs "1969 until 1974" → F1 = 0).
- Results are specific to the reader model and decoding settings listed above.
