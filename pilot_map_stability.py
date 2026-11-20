"""
Pilot: performance map + stability of a reader LLM under prompt compression.
Designed for Google Colab (free T4 GPU). See README section "Pilot on Colab".

For each HotpotQA query and each compression level we:
  1. compress the CONTEXT (documents) with LLMLingua-2; instruction and question stay intact,
  2. ask the reader LLM the question:
       - 1 greedy answer        (deterministic: same input -> same output)
       - K sampled answers      (temperature > 0: measures run-to-run variability)
  3. score each answer against the gold answer (Exact Match, F1 - official HotpotQA style),
  4. compute the probability the reader assigns to the gold answer (continuous score),
  5. record the ACTUAL compression ratio (LLMLingua-2 overshoots the requested rate).

Outputs:
  - pilot_results.json : one row per (query, compression level), everything kept
  - pilot_results.csv  : same, flat table (easy to open in pandas / Excel)
  - printed summary    : the performance map (mean score per level) and stability numbers

These rows are exactly the raw material of the supervisor's plan:
  score per (query, ratio)  ->  map  ->  labels  ->  estimator (embedding + regression).
"""
import argparse
import csv
import json
import math
import re
import string
import time
from collections import Counter

import numpy as np
import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

# Neutral wording: an earlier version ("a few words, or yes/no") biased the reader towards
# answering "No" even to non yes/no questions.
INSTRUCTION = ("Answer the question based on the documents below. "
               "Give only the answer, as a short phrase taken from the documents, without explanation. "
               "Only if it is a yes/no question, answer yes or no.")


# ----------------------------------------------------------------------------- scoring
def normalize_answer(s: str) -> str:
    """Official HotpotQA/SQuAD normalization: lowercase, drop punctuation, articles, extra spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """Token-overlap F1, with the official yes/no special case."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}:
        if p != g:
            return 0.0
    p_toks, g_toks = p.split(), g.split()
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision, recall = n_same / len(p_toks), n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


# ----------------------------------------------------------------------------- reader
class Reader:
    def __init__(self, model_name: str, device: str, dtype: str = "float16", load_in_4bit: bool = False,
                 multi_gpu: bool = False):
        # float16 by default: a Kaggle diagnostic (5 queries x 4 rates) gave identical answers
        # in float32 and float16 for Qwen2.5-3B, at half the memory.
        # multi_gpu: split the model over all visible GPUs (Kaggle gives 2 x T4), e.g. a 7B in
        # float16 (~15 GB) that does not fit on a single 15 GB T4. Inputs go to the first GPU.
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = getattr(torch, dtype) if device == "cuda" else torch.float32
        kwargs = {"torch_dtype": torch_dtype, "device_map": "auto" if multi_gpu else device}
        if load_in_4bit:  # e.g. a 7B model on a 16 GB T4
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()

    def prefix_ids(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False).input_ids)

    @torch.no_grad()
    def answers(self, prompt: str, n_samples: int, temperature: float, max_new_tokens: int, seed: int):
        """Returns (greedy_answer, [sampled answers])."""
        ids = self.prefix_ids(prompt)
        dec = lambda out: [self.tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip() for o in out]
        greedy = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=self.tok.eos_token_id)
        # Sampled answers are generated one at a time (not num_return_sequences=K) to keep the
        # GPU memory peak low: a 3B float32 model leaves only ~2 GB free on a 16 GB T4.
        sampled = []
        for k in range(n_samples):
            torch.manual_seed(seed * 1000 + k)
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                                      temperature=temperature, top_p=1.0,
                                      pad_token_id=self.tok.eos_token_id)
            sampled.extend(dec(out))
        return dec(greedy)[0], sampled

    @torch.no_grad()
    def _answer_logprobs(self, prefix, answer: str):
        ans = self.tok(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        ids = torch.cat([prefix, ans], dim=1)
        hidden = self.model.model(ids).last_hidden_state[0, prefix.shape[1] - 1: -1]
        logprobs = torch.log_softmax(self.model.lm_head(hidden).float(), dim=-1)
        # .to(logprobs.device): with --multi-gpu the output layer may sit on another GPU
        return logprobs.gather(1, ans[0].unsqueeze(1).to(logprobs.device)).squeeze(1)

    def gold_probability(self, prompt: str, gold: str):
        """Teacher-forced probability of the gold answer. Returns (seq_prob, per_token_prob):
          - seq_prob: P(model writes exactly this answer) = product of token probabilities.
            This is the meaningful score: the per-token mean can be high even when the first
            token is very unlikely (seen on Kaggle: answer "United States", per-token
            probability of the gold "9,984" = 0.86).
          - per_token_prob: geometric mean per token (kept for comparison).
        Max over casing variants: the model writes "No" while HotpotQA stores "no".
        The output layer is applied only to answer positions to save GPU memory."""
        prefix = self.prefix_ids(prompt)
        variants = {gold, gold[:1].upper() + gold[1:], gold.lower()}
        lps = [self._answer_logprobs(prefix, v) for v in variants if v]
        return (max(math.exp(lp.sum().item()) for lp in lps),
                max(math.exp(lp.mean().item()) for lp in lps))


def build_prompt(context: str, question: str) -> str:
    return f"{INSTRUCTION}\n\n{context}\n\nQuestion: {question}"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--compressor", default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    ap.add_argument("--n-queries", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0, help="first HotpotQA validation index to use")
    ap.add_argument("--screen-file", default=None,
                    help="JSON of a previous rate-1.0 run: keep only queries answered correctly at 1x")
    ap.add_argument("--screen-min-f1", type=float, default=0.5,
                    help="F1 threshold at 1x for --screen-file (default 0.5)")
    ap.add_argument("--rates", type=float, nargs="+", default=[1.0, 0.8, 0.5, 0.33],
                    help="LLMLingua-2 target rates (fraction of tokens kept). 1.0 = no compression.")
    ap.add_argument("--samples", type=int, default=5, help="sampled answers per (query, rate)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"],
                    help="reader precision on GPU")
    ap.add_argument("--load-in-4bit", action="store_true", help="4-bit quantized reader (bitsandbytes)")
    ap.add_argument("--multi-gpu", action="store_true", help="split the reader over all visible GPUs")
    ap.add_argument("--out", default="pilot_results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | reader: {args.reader} | compressor: {args.compressor}")

    # "hotpotqa/hotpot_qa" is the official parquet version: works with any `datasets` version
    # (the old "hotpot_qa" loading script is not supported by datasets>=4, e.g. on Colab).
    if args.screen_file:
        # Keep only queries the reader answers correctly WITHOUT compression in a previous
        # (cheap, rate-1.0-only) screening run: queries already wrong at 1x carry no
        # information about the tolerable compression ratio.
        screen = json.load(open(args.screen_file, encoding="utf-8"))["rows"]
        ids = sorted({r["query_index"] for r in screen
                      if r["target_rate"] == 1.0 and r["greedy_f1"] >= args.screen_min_f1})
        ids = ids[: args.n_queries]
        print(f"Screening file {args.screen_file}: {len(ids)} queries kept (F1 at 1x >= {args.screen_min_f1})")
        ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").select(ids)
    else:
        ids = list(range(args.offset, args.offset + args.n_queries))
        ds = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor",
                                   split=f"validation[{args.offset}:{args.offset + args.n_queries}]")

    # ---------------------------------------------------------------- phase 1: compress everything
    # Done first, then the compressor is freed, so the reader can use the whole GPU (float32).
    import gc
    from llmlingua import PromptCompressor
    t_start = time.time()
    compressor = PromptCompressor(model_name=args.compressor, use_llmlingua2=True, device_map=device)
    contexts = {}  # (query_idx, rate) -> context text
    for qi, s in enumerate(ds):
        docs = [f"Document [{t}]: {' '.join(sents)}"
                for t, sents in zip(s["context"]["title"], s["context"]["sentences"])]
        for rate in args.rates:
            if rate >= 1.0:
                contexts[(qi, rate)] = "\n\n".join(docs)
            else:
                res = compressor.compress_prompt(context=docs, rate=rate, use_token_level_filter=True)
                contexts[(qi, rate)] = res["compressed_prompt"]
    del compressor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"Compression done for {len(ds)} queries x {len(args.rates)} rates in {time.time() - t_start:.0f}s")

    # ---------------------------------------------------------------- phase 2: reader answers
    reader = Reader(args.reader, device, args.dtype, args.load_in_4bit, args.multi_gpu)
    rows = []
    for qi, s in enumerate(ds):
        full_ctx_tokens = reader.n_tokens(contexts[(qi, 1.0)]) if 1.0 in args.rates else None

        for rate in args.rates:
            context = contexts[(qi, rate)]
            ctx_tokens = reader.n_tokens(context)
            prompt = build_prompt(context, s["question"])
            if qi == 0 and rate == args.rates[0]:
                print("\n----- EXAMPLE PROMPT SENT TO THE READER (query 0) -----")
                print(prompt[:800] + "\n[...]\n" + prompt[-400:])
                print("-------------------------------------------------------\n")

            gold_seq_prob, gold_tok_prob = reader.gold_probability(prompt, s["answer"])
            greedy, sampled = reader.answers(prompt, args.samples, args.temperature,
                                             args.max_new_tokens, seed=args.seed + qi)
            sampled_f1 = [f1_score(a, s["answer"]) for a in sampled]
            sampled_em = [exact_match(a, s["answer"]) for a in sampled]

            rows.append({
                "query_index": ids[qi],
                "id": s["id"],
                "type": s["type"],
                "question": s["question"],
                "gold": s["answer"],
                "target_rate": rate,
                "context_tokens": ctx_tokens,
                "full_context_tokens": full_ctx_tokens,
                "actual_ratio": round(full_ctx_tokens / max(ctx_tokens, 1), 3) if full_ctx_tokens else None,
                "greedy_answer": greedy,
                "greedy_em": exact_match(greedy, s["answer"]),
                "greedy_f1": round(f1_score(greedy, s["answer"]), 4),
                "sampled_answers": sampled,
                "sampled_em": sampled_em,
                "sampled_f1": [round(x, 4) for x in sampled_f1],
                "sampled_f1_mean": round(float(np.mean(sampled_f1)), 4) if sampled else None,
                "sampled_f1_std": round(float(np.std(sampled_f1)), 4) if sampled else None,
                "gold_prob": round(gold_seq_prob, 5),          # P(exact gold answer)
                "gold_prob_per_token": round(gold_tok_prob, 5),  # geometric mean per token
            })
        print(f"[{qi + 1}/{len(ds)}] {time.time() - t_start:6.0f}s  {s['question'][:70]}")
        # Save after every query so a crash (e.g. GPU out of memory) never loses finished work.
        with open(f"{args.out}.json", "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "rows": rows}, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------- save
    flat_keys = [k for k in rows[0] if not isinstance(rows[0][k], list)]
    with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 100)
    print("PERFORMANCE MAP (mean over queries)")
    print("=" * 100)
    print(f"{'target':>6} | {'actual ratio':>12} | {'greedy EM':>9} | {'greedy F1':>9} | "
          f"{'sampled F1':>10} | {'gold prob':>9} | {'F1 std across runs':>18} | {'queries that flip':>17}")
    print("-" * 100)
    for rate in args.rates:
        r = [x for x in rows if x["target_rate"] == rate]
        flips = sum(1 for x in r if x["sampled_em"] and len(set(x["sampled_em"])) > 1)
        print(f"{rate:>6.2f} | {np.mean([x['actual_ratio'] for x in r]):>11.2f}x | "
              f"{np.mean([x['greedy_em'] for x in r]):>9.2f} | {np.mean([x['greedy_f1'] for x in r]):>9.2f} | "
              f"{np.mean([x['sampled_f1_mean'] for x in r]) if args.samples else float('nan'):>10.2f} | "
              f"{np.mean([x['gold_prob'] for x in r]):>9.3f} | "
              f"{np.mean([x['sampled_f1_std'] for x in r]) if args.samples else float('nan'):>18.3f} | "
              f"{flips:>8}/{len(r):<8}")
    print("\nHow to read it:")
    print(" - 'F1 std across runs' = average run-to-run spread of the score for the SAME query and SAME prompt.")
    print("   If it is comparable to the drop caused by compression, single runs cannot measure degradation.")
    print(" - 'queries that flip' = queries whose sampled answers are sometimes right, sometimes wrong (EM).")
    print(" - Greedy decoding is deterministic: its run-to-run spread is 0 by construction.")
    print(f"\nSaved {args.out}.json and {args.out}.csv  (total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
