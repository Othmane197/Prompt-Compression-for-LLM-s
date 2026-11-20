# Adaptive Prompt Compression for LLM Inference

Research project — CentraleSupélec × Manifold Technology (supervisor: Yannick Le Cacheux), 2026–2027.

**Question:** how much can a given LLM prompt (instructions + context + question) be compressed
before the LLM's answer degrades — and can we predict that compression ratio per prompt?

**Status:** pilot done (20 HotpotQA queries, 4 ratios, Qwen2.5-3B and 7B readers) — see RESULTS.md.

**Next steps**
1. Fix the target reader (labels are reader-specific).
2. Screening run: rate 1.0 only on ~600 queries; keep queries answered correctly without compression.
3. Dataset run: ~300 screened queries × 8 compression levels, greedy answers only
   (run-to-run variability was measured as small).
4. Labels ("max safe ratio", first-failure rule) and estimator baseline: query embedding + Ridge,
   compared to a constant-ratio baseline, train/test split by query.

## Results

See **[RESULTS.md](RESULTS.md)**. In short (pilot, 20 HotpotQA queries, Qwen2.5-3B reader):
run-to-run variability of the reader is small, but per-query performance is **not monotonic**
in the compression ratio (compression sometimes helps), and the average degradation is not yet
statistically established at this sample size.

## Repository

| Path | Content |
|---|---|
| `pilot_map_stability.py` | Main experiment: compress each query at several rates (LLMLingua-2), let a reader LLM answer (greedy + sampled), score EM/F1 and gold-answer probability, record actual ratios. |
| `plot_results.py` | Figures (mean map, query × ratio heatmap, per-query curves) and comparison of label rules for the "max safe ratio". `python plot_results.py [--greedy] [--show]` |
| `kaggle/pilot/` | Kaggle runner (T4 GPU) for the pilot. Edit `RUNS` in `run_pilot.py` to change the experiment. |
| `benchmark_hotpotqa.py` | Earlier compression-only benchmark (no reader, lexical proxies). |
| `results/` | Raw JSON outputs of all runs reported in RESULTS.md. |
| `RESULTS.md` | Measured results, setups and limitations. |

## How to run

**On Kaggle (recommended, free GPU)** — from this folder, with the Kaggle CLI logged in:
```bash
kaggle kernels push -p kaggle/pilot          # starts a run (code is cloned from this GitHub repo)
kaggle kernels status <user>/prompt-compression-pilot
kaggle kernels output <user>/prompt-compression-pilot -p kaggle_outputs/run
```
The runner clones this repository, so push code changes to GitHub before launching.

**On Colab** — GPU runtime (T4), then:
```python
!git clone https://github.com/Othmane197/Prompt-Compression-for-LLM-s.git
%cd Prompt-Compression-for-LLM-s
!pip install -q llmlingua datasets accelerate
!python pilot_map_stability.py --n-queries 20
```

**Locally:** `pip install -r requirements.txt`, then `python pilot_map_stability.py --help`.

## References

- Jiang et al., *LLMLingua*, EMNLP 2023.
- Pan et al., *LLMLingua-2*, Findings of ACL 2024.
- Yang et al., *HotpotQA*, EMNLP 2018.
