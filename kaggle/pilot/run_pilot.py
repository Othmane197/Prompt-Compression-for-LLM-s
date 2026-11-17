"""
Kaggle runner for pilot_map_stability.py.
Pushed with `kaggle kernels push -p kaggle/pilot`; results land in /kaggle/working
and are fetched with `kaggle kernels output`.

Edit RUNS below to change what is executed (e.g. quick test vs full pilot).
"""
import subprocess
import sys

# Each entry is one run of pilot_map_stability.py (executed in order).
RUNS = [
    ["--n-queries", "5", "--dtype", "float32", "--out", "/kaggle/working/diag_fp32"],
    ["--n-queries", "5", "--dtype", "float16", "--out", "/kaggle/working/diag_fp16"],
]
REPO = "https://github.com/Othmane197/Prompt-Compression-for-LLM-s.git"


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


sh([sys.executable, "-m", "pip", "install", "-q", "llmlingua"])
sh(["git", "clone", "--depth", "1", REPO, "/kaggle/working/repo"])
sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
for args in RUNS:
    sh([sys.executable, "/kaggle/working/repo/pilot_map_stability.py", *args])
