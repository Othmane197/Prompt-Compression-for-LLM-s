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
# Current step: stability pilot (20 queries x 4 rates x greedy + 5 sampled answers).
RUNS = [
    ["--n-queries", "20", "--samples", "5", "--reader", "Qwen/Qwen2.5-3B-Instruct",
     "--out", "/kaggle/working/pilot_3b"],
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
