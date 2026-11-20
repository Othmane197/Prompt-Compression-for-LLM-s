# Adaptive Prompt Compression for LLM Inference

Research project — CentraleSupélec × Manifold Technology (supervisor: Yannick Le Cacheux), 2026–2027.

**Question:** how much can a given LLM prompt (instructions + context + question) be compressed
before the LLM's answer degrades — and can we predict that compression ratio per prompt?

**Status:** dataset of 300 HotpotQA queries × 8 compression ratios (Qwen2.5-7B reader) and first
estimator baselines done — see RESULTS.md §0.

**Next steps**
1. Make the estimator beat the simple baselines: Ridge on a generic embedding ties with a constant
   ratio, while the question type alone beats it. Try formulation (b) ((query, ratio) → OK / not OK,
   classifier), question-only embeddings, tuned regularization, simple features (context length).
2. Decide with the supervisor: target reader, label rule (64/300 curves recover after a failure),
   acceptable loss.

## Results

See **[RESULTS.md](RESULTS.md)**. In short (300 HotpotQA queries, Qwen2.5-7B reader): mean F1 falls
from 0.93 (1×, screened queries) to 0.60 at ~2.9× and 0.35 at ~8×; the tolerable ratio varies
widely across queries (1× to >8×). A constant ratio keeps 70% of answers at 2.0×; predicting by
question type gives 73% at 2.6×; Ridge on a query embedding does not beat the constant; an oracle
would reach 100% at 3.85×.

## Repository

| Path | Content |
|---|---|
| `pilot_map_stability.py` | Main experiment: compress each query at several rates (LLMLingua-2), let a reader LLM answer (greedy + sampled), score EM/F1 and gold-answer probability, record actual ratios. |
| `plot_results.py` | Figures (mean map, query × ratio heatmap, per-query curves) and comparison of label rules for the "max safe ratio". `python plot_results.py [--greedy] [--show]` |
| `train_estimator.py` | Baseline estimator: labels (max safe ratio, first-failure rule), query embedding (bge-small) + Ridge vs constant ratio, cross-validation by query; reports safety and mean compression applied. |
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
