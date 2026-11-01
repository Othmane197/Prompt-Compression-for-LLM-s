import os
import torch
import transformers
import datasets
import llmlingua
import spacy

print("=" * 60)
print("       AI LAB ENVIRONMENT VERIFICATION (DRIVE E:)")
print("=" * 60)

# Check PyTorch & CUDA
print(f"[OK] PyTorch version: {torch.__version__}")
cuda_avail = torch.cuda.is_available()
print(f"[OK] CUDA Available:  {cuda_avail}")
if cuda_avail:
    print(f"[OK] GPU Detected:    {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[OK] GPU VRAM:        {vram_gb:.2f} GB")

# Check HuggingFace & LLMLingua
print(f"[OK] Transformers:    {transformers.__version__}")
print(f"[OK] Datasets:        {datasets.__version__}")
print(f"[OK] LLMLingua:       {llmlingua.__version__}")

# Check spaCy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Testing spaCy lemmatization: running dogs went home.")
lemmas = [t.lemma_ for t in doc]
print(f"[OK] spaCy Model:     en_core_web_sm loaded successfully")
print(f"     Lemmas test:     {lemmas}")

# Check Hugging Face Cache on E:
hf_home = os.environ.get("HF_HOME", "Default (~/.cache)")
print(f"[OK] HF Cache Dir:    {hf_home}")
print("=" * 60)
print("All dependencies are ready for your prompt compression project!")
print("=" * 60)
