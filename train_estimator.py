"""
Baseline estimator of the compression ratio a query tolerates (supervisor's plan, step 4).

Input : one or more run JSONs from pilot_map_stability.py (several ratios per query).
Steps :
  1. Labels. For each query answered correctly at 1x (F1 >= --min-baseline), the
     "max safe ratio" = largest measured ratio before the FIRST ratio whose F1 falls below
     F1(1x) - tolerance (conservative "first failure" rule, no smoothing).
  2. Features. Query embedding with a small pretrained encoder (default BAAI/bge-small-en-v1.5):
     [embedding(question), mean of embedding(each document)]  -> 2 x 384 numbers.
  3. Models (formulation (a) of the supervisor: query -> max safe ratio), target = log2(ratio):
       - constant baseline : always predict the median training label
       - Ridge regression on the embedding
  4. Evaluation with cross-validation split BY QUERY (a query is never in train and test).
     Besides the regression error, we report what matters in practice: compress each test
     query at the predicted ratio (snapped down to the nearest measured ratio) and check
       - safety      : share of test queries whose answer stays OK
       - compression : mean ratio actually applied

Usage:
    python train_estimator.py results/pilot_7b.json
    python train_estimator.py runs/dataset_7b.json --folds 5 --alpha 10
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


# ----------------------------------------------------------------------------- labels
def build_labels(rows, tolerance, min_baseline):
    by_q = {}
    for r in rows:
        by_q.setdefault(r["query_index"], []).append(r)
    data = []
    for q, rs in sorted(by_q.items()):
        rs = sorted(rs, key=lambda r: r["actual_ratio"])
        if rs[0]["target_rate"] != 1.0:
            continue  # need the uncompressed reference
        base = rs[0]["greedy_f1"]
        if base < min_baseline:
            continue
        ratios = [r["actual_ratio"] for r in rs]
        ok = [r["greedy_f1"] >= base - tolerance for r in rs]
        label = ratios[0]
        for ratio, o in zip(ratios, ok):
            if not o:
                break
            label = ratio
        data.append({"query_index": q, "question": rs[0]["question"],
                     "ratios": ratios, "ok": ok, "label": label})
    return data


# ----------------------------------------------------------------------------- features
def load_documents(indices):
    import datasets
    try:
        ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    except Exception:
        ds = datasets.load_dataset("hotpot_qa", "distractor", split="validation")
    sub = ds.select(indices)
    return [[f"{t}: {' '.join(s)}" for t, s in zip(x["context"]["title"], x["context"]["sentences"])]
            for x in sub]


class Embedder:
    def __init__(self, name):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    @torch.no_grad()
    def __call__(self, texts):
        out = []
        for i in range(0, len(texts), 16):
            batch = self.tok(texts[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt")
            cls = self.model(**batch).last_hidden_state[:, 0]          # CLS pooling (bge models)
            out.append(torch.nn.functional.normalize(cls, dim=-1))
        return torch.cat(out).numpy()


def build_features(data, embedder_name):
    emb = Embedder(embedder_name)
    docs = load_documents([d["query_index"] for d in data])
    q_vec = emb([d["question"] for d in data])
    d_vec = np.stack([emb(ds).mean(axis=0) for ds in docs])
    return np.hstack([q_vec, d_vec])


# ----------------------------------------------------------------------------- evaluation
def apply_ratio(d, predicted_ratio):
    """Snap the prediction down to the largest measured ratio <= prediction, report outcome."""
    idx = max([i for i, r in enumerate(d["ratios"]) if r <= predicted_ratio] or [0])
    return d["ok"][idx], d["ratios"][idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run JSON(s) with several ratios per query")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--min-baseline", type=float, default=0.5)
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization strength")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [r for p in args.paths for r in json.load(open(p, encoding="utf-8"))["rows"]]
    data = build_labels(rows, args.tolerance, args.min_baseline)
    n = len(data)
    y = np.log2([d["label"] for d in data])
    print(f"{n} labelled queries (F1 at 1x >= {args.min_baseline}); "
          f"label = max safe ratio, median {2 ** np.median(y):.2f}x, range {2 ** y.min():.2f}x - {2 ** y.max():.2f}x")
    if n < 2 * args.folds:
        print(f"WARNING: only {n} queries for {args.folds}-fold CV - numbers below are not meaningful.")
    X = build_features(data, args.embedder)
    print(f"Features: {X.shape[1]} dimensions ({args.embedder})")

    k = min(args.folds, n)
    pred_const, pred_ridge = np.zeros(n), np.zeros(n)
    for train, test in KFold(n_splits=k, shuffle=True, random_state=args.seed).split(X):
        pred_const[test] = np.median(y[train])
        model = Ridge(alpha=args.alpha).fit(X[train], y[train])
        pred_ridge[test] = model.predict(X[test])

    print(f"\nCross-validation by query ({k} folds). Target = log2(max safe ratio).")
    print(f"{'model':<22} | {'MAE (log2)':>10} | {'safety':>7} | {'mean ratio applied':>18}")
    print("-" * 66)
    for name, pred in [("constant (median)", pred_const), ("Ridge on embedding", pred_ridge)]:
        mae = np.mean(np.abs(pred - y))
        outcomes = [apply_ratio(d, 2 ** p) for d, p in zip(data, pred)]
        safety = np.mean([o for o, _ in outcomes])
        mean_ratio = np.mean([r for _, r in outcomes])
        print(f"{name:<22} | {mae:>10.3f} | {safety:>6.0%} | {mean_ratio:>17.2f}x")
    oracle = np.mean([d["label"] for d in data])
    print(f"{'oracle (true label)':<22} | {0:>10.3f} | {1:>6.0%} | {oracle:>17.2f}x")
    print("\nsafety = share of test queries whose answer stays within tolerance at the applied ratio;"
          "\nthe oracle is the best achievable (each query compressed exactly to its own label).")


if __name__ == "__main__":
    main()
