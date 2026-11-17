"""
Visualize a pilot run (results/pilot_3b.json by default) and compare label rules.

Usage (from the project folder, venv activated):
    python plot_results.py                       # figures -> figures/, labels table printed
    python plot_results.py results/other.json    # another run
    python plot_results.py --show                # also open the figures in a window

Figures:
  figures/1_mean_map.png       mean score vs actual compression ratio
  figures/2_heatmap.png        score of every query at every ratio (the "map")
  figures/3_per_query.png      one small curve per query (non-monotonic ones flagged)

Label rules (printed table) - "max safe ratio" for each query that is answered
correctly without compression. A ratio is "OK" if F1 >= F1(1x) - tolerance.
  first_failure : largest ratio before the FIRST non-OK ratio (conservative, no smoothing)
  last_success  : largest OK ratio, ignoring failures before it (optimistic)
  monotone_fit  : fit the closest non-increasing curve (isotonic regression), then
                  largest ratio where the fitted curve is still OK (statistical middle ground)
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated reference palette (light mode) - see dataviz skill palette
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def load(path):
    data = json.load(open(path, encoding="utf-8"))
    rows = data["rows"]
    rates = sorted({r["target_rate"] for r in rows}, reverse=True)  # 1.0 first
    queries = sorted({r["query_index"] for r in rows})
    by = {(r["query_index"], r["target_rate"]): r for r in rows}
    return data["config"], rows, rates, queries, by


USE_GREEDY = False  # set by --greedy


def score(r):
    """Per-cell score: mean of sampled answers if available (unless --greedy), else the greedy answer."""
    if USE_GREEDY or r.get("sampled_f1_mean") is None:
        return r["greedy_f1"]
    return r["sampled_f1_mean"]


def isotonic_decreasing(y):
    """Pool-adjacent-violators: closest non-increasing sequence to y (least squares)."""
    blocks = [[v, 1] for v in y]  # [mean, size]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:  # violation: later value higher
            m = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [m, blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for m, n in blocks:
        out += [m] * n
    return out


def labels_for(curve, ratios, tol):
    """curve: scores ordered by increasing ratio (curve[0] = 1x). Returns the 3 labels."""
    ok = [c >= curve[0] - tol for c in curve]
    first_failure = ratios[0]
    for r, o in zip(ratios, ok):
        if not o:
            break
        first_failure = r
    last_success = max(r for r, o in zip(ratios, ok) if o)
    fit = isotonic_decreasing(curve)
    monotone = max(r for r, f in zip(ratios, fit) if f >= curve[0] - tol)
    return first_failure, last_success, monotone, fit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="results/pilot_3b.json")
    ap.add_argument("--tolerance", type=float, default=0.05, help="acceptable F1 loss vs 1x (default 0.05)")
    ap.add_argument("--min-baseline", type=float, default=0.5,
                    help="queries with F1(1x) below this are excluded from labels (default 0.5)")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--greedy", action="store_true", help="score cells with the single greedy answer")
    args = ap.parse_args()
    global USE_GREEDY
    USE_GREEDY = args.greedy
    if args.greedy and args.out == "figures":
        args.out = "figures/greedy"

    cfg, rows, rates, queries, by = load(args.path)
    os.makedirs(args.out, exist_ok=True)
    mean_ratio = [np.mean([by[(q, t)]["actual_ratio"] for q in queries]) for t in rates]
    print(f"Run: reader={cfg['reader']} | {len(queries)} queries | rates={rates}")

    # ---------------------------------------------------------------- 1. mean map
    fig, ax = plt.subplots(figsize=(6.4, 4))
    greedy = [np.mean([by[(q, t)]["greedy_f1"] for q in queries]) for t in rates]
    sampled = [np.mean([score(by[(q, t)]) for q in queries]) for t in rates]
    ax.plot(mean_ratio, greedy, color=BLUE, lw=2, marker="o", ms=7, label="Greedy F1")
    ax.plot(mean_ratio, sampled, color=ORANGE, lw=2, marker="o", ms=7, label="Sampled F1 (mean of runs)")
    # direct labels, nudged apart when the two end points are close
    hi, lo = (greedy[-1], sampled[-1]) if greedy[-1] >= sampled[-1] else (sampled[-1], greedy[-1])
    gap = max(0.0, 0.06 - (hi - lo)) / 2
    g_y = greedy[-1] + (gap if greedy[-1] >= sampled[-1] else -gap)
    s_y = sampled[-1] + (gap if sampled[-1] > greedy[-1] else -gap)
    ax.text(mean_ratio[-1] + 0.1, g_y, "Greedy", color=INK_2, va="center")
    ax.text(mean_ratio[-1] + 0.1, s_y, "Sampled", color=INK_2, va="center")
    ax.set_xlabel("Actual compression ratio (×)")
    ax.set_ylabel("Mean F1 over queries")
    ax.set_ylim(0, 1)
    ax.set_xlim(0.8, max(mean_ratio) + 0.9)
    ax.set_title(f"Answer quality vs compression ({len(queries)} queries)", loc="left", color=INK)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{args.out}/1_mean_map.png", dpi=150)

    # ---------------------------------------------------------------- 2. heatmap (the map)
    grid = np.array([[score(by[(q, t)]) for t in rates] for q in queries])
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(queries) + 1.4))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(rates)), [f"{r:.1f}×" for r in mean_ratio])
    ax.set_yticks(range(len(queries)), [f"Q{q}" for q in queries])
    ax.grid(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else INK)
    ax.set_xlabel("Actual compression ratio (mean over queries)")
    ax.set_title("F1 per query and ratio", loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.05, label="F1")
    fig.tight_layout()
    fig.savefig(f"{args.out}/2_heatmap.png", dpi=150)

    # ---------------------------------------------------------------- 3. per-query curves
    n = len(queries)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 1.9 * nrows), sharex=True, sharey=True)
    for k, q in enumerate(queries):
        ax = axes.flat[k]
        xs = [by[(q, t)]["actual_ratio"] for t in rates]
        ys = [score(by[(q, t)]) for t in rates]
        non_mono = any(ys[i + 1] > ys[i] + 0.2 for i in range(len(ys) - 1))
        ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Q{q}" + ("  · goes back up" if non_mono else ""), fontsize=9, loc="left",
                     color=ORANGE if non_mono else INK)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.supxlabel("Actual compression ratio (×)", color=INK_2)
    fig.supylabel("F1", color=INK_2)
    fig.suptitle("Per-query curves (title in orange: score rises again after a drop)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(f"{args.out}/3_per_query.png", dpi=150)

    # ---------------------------------------------------------------- labels
    print(f"\nLABELS ('max safe ratio'), tolerance = {args.tolerance} F1, "
          f"queries with F1(1x) < {args.min_baseline} excluded")
    print(f"{'query':>5} | {'curve (F1 by ratio)':<28} | {'first_failure':>13} | {'last_success':>12} | {'monotone_fit':>12}")
    print("-" * 85)
    kept, disagree = 0, 0
    for q in queries:
        curve = [score(by[(q, t)]) for t in rates]
        ratios = [by[(q, t)]["actual_ratio"] for t in rates]
        if curve[0] < args.min_baseline:
            continue
        kept += 1
        ff, ls, mf, _ = labels_for(curve, ratios, args.tolerance)
        if len({ff, ls, mf}) > 1:
            disagree += 1
        flag = "  <- rules disagree" if len({ff, ls, mf}) > 1 else ""
        print(f"{'Q' + str(q):>5} | {' '.join(f'{c:.2f}' for c in curve):<28} | {ff:>12.2f}× | {ls:>11.2f}× | {mf:>11.2f}×{flag}")
    print(f"\n{kept} usable queries; the 3 rules disagree on {disagree} of them.")
    print(f"Figures saved in {args.out}/")
    if args.show:
        matplotlib.use("TkAgg", force=True)
        for f in sorted(os.listdir(args.out)):
            img = plt.imread(os.path.join(args.out, f))
            plt.figure(figsize=(9, 6))
            plt.imshow(img)
            plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
