"""
Deletion test: which pieces of a prompt actually matter to the AI's answer?

Idea ("cook the recipe without each ingredient"):
  1. Build a RAG prompt for one HotpotQA question: instruction + 10 documents + question.
  2. Ask a small reader LLM the question with the FULL prompt and measure how confident
     it is in the correct answer.
  3. Remove ONE piece at a time (each document, the instruction) and measure again.
  4. Sensitivity of a piece = how much the confidence drops when that piece is removed.

We also compare against LLMLingua-2 uniform compression (rate 0.5), which cuts every
piece equally without looking at the question.

This is step 1 of the proposed pipeline: these sensitivity scores are the "labels"
a fast predictor would later be trained to estimate without doing any deletion.

Usage:
  python sensitivity_deletion_test.py            # sample 8 (Lewiston Maineiacs arena)
  python sensitivity_deletion_test.py --sample 2 # another HotpotQA validation sample
"""
import os
import re
import json
import math
import time
import argparse

os.environ.setdefault("HF_HOME", r"E:\AI_Lab_Environment\hf_cache")  # only if not already set system-wide

import torch
import datasets
import struct
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

READER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INSTRUCTION = "Answer the question using the documents below. Reply with the answer only, as briefly as possible."


LLMLINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
# This machine has two Hugging Face caches (system HF_HOME and the project's own folder).
EXTRA_CACHES = [r"E:\AI_Lab_Environment\hf_cache\hub", r"E:\cache\huggingface\hub"]


def find_local_model(repo_id):
    """Return a local snapshot path for repo_id from any known cache, else the repo id
    itself (which lets Hugging Face download it when online)."""
    from huggingface_hub import snapshot_download
    for cache in [None] + EXTRA_CACHES:
        try:
            return snapshot_download(repo_id, cache_dir=cache, local_files_only=True)
        except Exception:
            continue
    return repo_id


SAFETENSORS_DTYPES ={"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def stream_safetensors(path):
    """Yield (name, CPU tensor) one at a time from a .safetensors file without memory-mapping.
    File layout: 8-byte little-endian header size, JSON header, then raw tensor bytes."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            raw = bytearray(f.read(end - begin))
            yield name, torch.frombuffer(raw, dtype=SAFETENSORS_DTYPES[info["dtype"]]).reshape(info["shape"])


def build_prompt(instruction, docs, question):
    """Assemble the prompt text from its pieces (any piece can be None = removed)."""
    parts = []
    if instruction:
        parts.append(instruction)
    for title, body in docs:
        parts.append(f"Document [{title}]: {body}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


class Reader:
    """Small LLM that answers the question and scores the gold answer."""

    def __init__(self, device):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(READER_MODEL)
        dtype = torch.float16 if device == "cuda" else torch.float32
        # Low-RAM loading. transformers' default loader crashes on this machine (access
        # violation / MemoryError) because it needs ~2-3 GB of free system RAM. Instead we
        # build the empty model directly on the GPU and stream the weights in one tensor
        # at a time, so at most a few hundred MB sit in system RAM at once.
        weights_path = hf_hub_download(READER_MODEL, "model.safetensors")
        config = AutoConfig.from_pretrained(READER_MODEL)
        with torch.device(device):
            self.model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        params = dict(self.model.named_parameters())
        for name, tensor in stream_safetensors(weights_path):
            if name in params:
                params[name].data.copy_(tensor.to(device=device, dtype=dtype))
        self.model.tie_weights()  # Qwen2.5-0.5B shares input embeddings and output layer
        self.model.eval()

    def _prefix_ids(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def answer_confidence(self, prompt, gold_answer):
        """
        Teacher-forced score: feed prompt + gold answer, read the probability the model
        assigns to each answer token. Returns the geometric-mean per-token probability
        (0..1): "how confident is the model that this is the answer?"
        """
        prefix = self._prefix_ids(prompt)
        ans = self.tok(gold_answer, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        ids = torch.cat([prefix, ans], dim=1)
        # Run the transformer body, then apply the (huge, 152k-word) output layer only on the
        # positions that predict the answer tokens; doing it for all ~3k prompt positions
        # would need ~1 GB of GPU memory.
        hidden = self.model.model(ids).last_hidden_state[0, prefix.shape[1] - 1 : -1]
        logits = self.model.lm_head(hidden).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        token_lp = logprobs.gather(1, ans[0].unsqueeze(1)).squeeze(1)
        return math.exp(token_lp.mean().item())

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=16):
        prefix = self._prefix_ids(prompt)
        out = self.model.generate(prefix, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, prefix.shape[1]:], skip_special_tokens=True).strip()


def is_correct(generated, gold):
    """Lenient check: every word/number of the gold answer's key part appears in the output."""
    norm = lambda s: re.findall(r"\w+", s.lower().replace(",", ""))
    gold_words = norm(gold)
    gen_words = set(norm(generated))
    # accept if all gold words appear, or the first gold word (e.g. the number) appears
    return all(w in gen_words for w in gold_words) or (gold_words and gold_words[0] in gen_words)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=8, help="1-based index in HotpotQA validation (default 8)")
    parser.add_argument("--no-llmlingua", action="store_true", help="skip the LLMLingua-2 comparison")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = datasets.load_dataset("hotpot_qa", "distractor", split=f"validation[{args.sample - 1}:{args.sample}]")
    s = ds[0]
    question, gold = s["question"], s["answer"]
    docs = [(t, " ".join(sents)) for t, sents in zip(s["context"]["title"], s["context"]["sentences"])]
    gold_titles = set(s["supporting_facts"]["title"])
    del ds

    # Compress with LLMLingua-2 FIRST and unload it before loading the reader LLM:
    # on low-RAM machines, having both loaded at once crashes.
    llmlingua_prompt = None
    if not args.no_llmlingua:
        import gc
        from llmlingua import PromptCompressor
        print("Loading LLMLingua-2 for the uniform-compression comparison...")
        comp = PromptCompressor(model_name=find_local_model(LLMLINGUA_MODEL),
                                use_llmlingua2=True, device_map=device)
        doc_texts = [f"Document [{t}]: {b}" for t, b in docs]
        res = comp.compress_prompt(context=doc_texts, rate=0.5, use_token_level_filter=True)
        llmlingua_prompt = f"{INSTRUCTION}\n\n{res['compressed_prompt']}\n\nQuestion: {question}"
        del comp, res
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"Loading reader LLM {READER_MODEL} on {device}...")
    reader = Reader(device)

    print("=" * 90)
    print(f"Question : {question}")
    print(f"Gold     : {gold}")
    print(f"Useful docs (according to the dataset): {sorted(gold_titles)}")
    print("=" * 90)

    rows = []

    def run(label, prompt, kind):
        t0 = time.time()
        conf = reader.answer_confidence(prompt, gold)
        gen = reader.generate(prompt)
        n_tok = len(reader.tok(prompt).input_ids)
        rows.append({"condition": label, "kind": kind, "prompt_tokens": n_tok,
                     "confidence": conf, "generated": gen, "correct": is_correct(gen, gold)})
        print(f"  done: {label:<45} ({time.time() - t0:.1f}s)")

    print("\nRunning conditions...")
    full_prompt = build_prompt(INSTRUCTION, docs, question)
    run("FULL PROMPT (nothing removed)", full_prompt, "reference")
    run("without: instruction", build_prompt(None, docs, question), "deletion")
    for i, (title, _) in enumerate(docs):
        tag = "useful" if title in gold_titles else "distractor"
        kept = docs[:i] + docs[i + 1:]
        run(f"without: [{title}] ({tag})", build_prompt(INSTRUCTION, kept, question), "deletion")
    run("ONLY useful docs (perfect selector)",
        build_prompt(INSTRUCTION, [d for d in docs if d[0] in gold_titles], question), "selection")

    if llmlingua_prompt is not None:
        run("LLMLingua-2 uniform, rate 0.5", llmlingua_prompt, "compression")

    # ---- Report ----
    ref = rows[0]["confidence"]
    print("\n" + "=" * 90)
    print("RESULTS  (confidence = how sure the AI is of the correct answer, 0-100%)")
    print("         (sensitivity = confidence lost when the piece is removed; big = important)")
    print("=" * 90)
    print(f"{'Condition':<48} {'Tokens':>6} {'Conf.':>7} {'Sensitivity':>12}  {'OK?':<4} AI answer")
    print("-" * 90)
    for r in rows:
        r["sensitivity"] = ref - r["confidence"]
        sens = f"{r['sensitivity'] * 100:+.1f} pts" if r["kind"] == "deletion" else ""
        print(f"{r['condition'][:48]:<48} {r['prompt_tokens']:>6} {r['confidence'] * 100:>6.1f}% "
              f"{sens:>12}  {'yes' if r['correct'] else 'NO':<4} {r['generated'][:40]!r}")

    deletions = sorted([r for r in rows if r["kind"] == "deletion"], key=lambda r: -r["sensitivity"])
    print("\nMost important pieces according to the deletion test:")
    for r in deletions[:3]:
        print(f"  {r['condition'].replace('without: ', ''):<50} {r['sensitivity'] * 100:+.1f} pts")

    out = f"sensitivity_results_sample{args.sample}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"question": question, "gold_answer": gold, "reader_model": READER_MODEL,
                   "useful_docs": sorted(gold_titles), "results": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
