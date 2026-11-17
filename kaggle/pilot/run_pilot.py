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
# Current step: reader choice. Uncompressed baseline (rate 1.0 only, greedy only) on 20 queries,
# Qwen2.5-3B (float16) vs Qwen2.5-7B (4-bit), after fixing the prompt and the gold-prob casing.
RUNS = [
    ["--n-queries", "20", "--rates", "1.0", "--samples", "0",
     "--reader", "Qwen/Qwen2.5-3B-Instruct", "--out", "/kaggle/working/baseline_3b"],
    ["--n-queries", "20", "--rates", "1.0", "--samples", "0",
     "--reader", "Qwen/Qwen2.5-7B-Instruct", "--load-in-4bit", "--out", "/kaggle/working/baseline_7b_4bit"],
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
