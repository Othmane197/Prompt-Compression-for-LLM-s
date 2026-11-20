"""
Kaggle runner for pilot_map_stability.py.
Pushed with `kaggle kernels push -p kaggle/pilot`; results land in /kaggle/working
and are fetched with `kaggle kernels output`.

Edit RUNS below to change what is executed (e.g. quick test vs full pilot).
"""
import os
import subprocess
import sys

REPO = "https://github.com/Othmane197/Prompt-Compression-for-LLM-s.git"

# Each entry is one run of pilot_map_stability.py (executed in order).
# History: (1) float32 vs float16 diagnostic -> identical answers; (2) uncompressed baseline on
# 20 queries: Qwen2.5-3B EM 0.40 / F1 0.50 vs Qwen2.5-7B-4bit EM 0.45 / F1 0.54 (not a
# significant gap, 2x slower) -> 3B kept.
# (3) stability pilot with the 3B -> results/pilot_3b.json.
# (4) same pilot with the full-precision 7B on 2 x T4 -> results/pilot_7b.json.
# Current step: screening with the 7B on a first batch of 100 queries (no compression, greedy),
# to measure runtime and the share of queries answered correctly at 1x.
RUNS = [
    ["--n-queries", "100", "--rates", "1.0", "--samples", "0",
     "--reader", "Qwen/Qwen2.5-7B-Instruct", "--multi-gpu", "--out", "/kaggle/working/screen_7b_100"],
]


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


sh([sys.executable, "-m", "pip", "install", "-q", "llmlingua", "bitsandbytes"])
sh(["git", "clone", "--depth", "1", REPO, "/kaggle/working/repo"])
sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # less memory fragmentation
for args in RUNS:
    # check=False: a failing run must not prevent the next one
    print("+", "pilot", " ".join(args), flush=True)
    subprocess.run([sys.executable, "/kaggle/working/repo/pilot_map_stability.py", *args])
    # Figures for this run, visible in the notebook's Output tab (a failure here is harmless).
    out = args[args.index("--out") + 1]
    if os.path.exists(out + ".json"):
        subprocess.run([sys.executable, "/kaggle/working/repo/plot_results.py", out + ".json",
                        "--out", out + "_figures"])
