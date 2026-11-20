# Results

Measured results only. Every number below can be recomputed from the JSON files in [`results/`](results/).
§0 uses 300 queries; §1–3 are 20-query (or 10-query) **pilots**, indicative only.

---

## 0. Dataset run: 300 queries × 8 compression levels, and the first estimator (main result)

**Setup:** HotpotQA (distractor, validation) · reader Qwen2.5-7B-Instruct (float16, 2 × T4) ·
compressor LLMLingua-2 `xlm-roberta-large-meetingbank` (documents only) · greedy answers only.
1. *Screening:* queries 0–449 at rate 1.0 (no compression); keep F1 ≥ 0.5 → 330/450 usable
   (`results/screen_7b_000_099.json`, `results/screen_7b_100_449.json`).
2. *Dataset:* the first 300 usable queries (239 bridge, 61 comparison) × 8 target rates
   (`results/dataset_7b.json`, 2,400 answers, ~3.5 h on Kaggle). Uncompressed answers are
   identical to the screening run for 300/300 queries (greedy decoding is reproducible).

| Target rate | 1.0 | 0.9 | 0.8 | 0.65 | 0.5 | 0.4 | 0.3 | 0.2 |
|---|---|---|---|---|---|---|---|---|
| Actual ratio (median) | 1.00× | 1.25× | 1.39× | 1.96× | 2.92× | 3.62× | 5.55× | 8.12× |
| Actual ratio (range) | – | 1.13–3.36 | 1.24–3.70 | 1.62–4.51 | 2.12–7.40 | 2.60–8.81 | 3.80–11.56 | 5.39–18.50 |
| Greedy EM | 0.79 | 0.69 | 0.64 | 0.61 | 0.44 | 0.41 | 0.32 | 0.23 |
| Greedy F1 | 0.93 | 0.83 | 0.79 | 0.76 | 0.60 | 0.54 | 0.45 | 0.35 |

F1 at 1× is high **by construction** (screened queries); the table measures degradation on queries
the reader can answer, not overall reader accuracy.

**Labels** (max safe ratio, first-failure rule, tolerance 0.05 F1): median 2.48×, range 1.00–14.37×.
38/300 queries already fail at the first compression level; 68/300 never fail (label = highest
ratio measured, i.e. **right-censored**). 64/300 queries fail at some level and recover at a
higher one, so the first-failure rule is conservative for them.
Comparison questions tolerate much more compression than bridge questions (median label 7.1×
vs 2.1×), including comparison questions with a span answer (7.4×, n = 35), so this is not just
yes/no guessing.

**Estimator** — formulation (a): query → max safe ratio. 5-fold cross-validation split by query;
the predicted ratio is snapped down to a measured ratio and applied to the test query.
Script: `python train_estimator.py results/dataset_7b.json`.

| Policy | MAE (log2) | Safety | Mean ratio applied |
|---|---|---|---|
| Constant (median training label) | 1.003 | 70% | 2.01× |
| Question type (median label per HotpotQA type) | 0.913 | 73% | 2.56× |
| Ridge on bge-small embedding (question + mean of documents) | 0.991 | 66% | 2.31× |
| Oracle (each query at its own label) | 0 | 100% | 3.85× |

For reference, one fixed target rate for all queries: rate 0.65 → 75% safe at 2.01×;
rate 0.5 → 55% safe at 2.99×.

**Findings**

1. **The Ridge-on-embedding baseline does not beat the constant.** Its MAE is equal (0.99 vs
   1.00) and it only moves along the safety/compression trade-off (66% at 2.31× vs 70% at 2.01×),
   which a fixed rate does as well.
2. **A one-bit feature does beat it:** predicting by question type gives +0.5× compression and
   +3 points of safety (stable over 3 CV seeds: 72–73% at 2.54–2.56× vs 69–71% at 2.01–2.03×).
   The safety gain is within noise at n = 300; the compression gain is clear. The type is a dataset
   annotation, but comparison questions are recognisable from their wording.
   → The signal exists; the generic embedding + Ridge setup fails to extract it.
3. **Large headroom:** the oracle reaches 3.85× at 100% safety; the best baseline 2.56× at 73%.

## Earlier pilots

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

- §0: 300 queries, one reader (Qwen2.5-7B), one dataset (HotpotQA), one compressor family; labels are
  reader-specific and right-censored at the highest measured ratio for 68/300 queries.
- §1–3: small samples (20 and 10 queries).
- Official EM/F1 normalization penalizes format differences ("1969-1974" vs "1969 until 1974" → F1 = 0).
- Results are specific to the reader model and decoding settings listed above.
