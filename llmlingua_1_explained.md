# Building LLMLingua-1 from Scratch: The Complete Beginner's Guide

> **Paper Title:** *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models* (EMNLP 2023)  
> **Core Concept:** Information Entropy & Perplexity-Guided Prompt Pruning  
> **Target Audience:** Anyone new to Natural Language Processing (NLP) who wants to understand and build prompt compression systems from the ground up.

---

## 1. The Core Problem: Why Do We Need Prompt Compression?

### 1.1 The Context Explosion
Modern Large Language Models (LLMs) like GPT-4, Claude, or LLaMA are rarely used with just a simple one-sentence question. Instead, we feed them massive amounts of context:
1. **In-Context Learning (ICL):** Giving the model 5 to 10 full examples of questions and answers before asking our question.
2. **Retrieval-Augmented Generation (RAG):** Searching a database of 20 long documents, pasting them all into the prompt, and telling the model: *"Answer based on these documents."*
3. **Conversational History:** Chatbots that carry over weeks of past dialogue.

These prompts can easily reach **4,000 to 30,000 tokens** (words and word pieces).

### 1.2 The Three Pain Points
Feeding huge prompts into an LLM causes three severe problems:
1. **Financial Cost:** Commercial APIs (like OpenAI) charge per token. A 10,000-token prompt costs 10x more than a 1,000-token prompt.
2. **Latency (Delay):** Before an LLM can generate its first word, it must process every single token in the prompt. Longer prompts mean longer waiting times.
3. **The "Lost-in-the-Middle" Phenomenon:** LLMs are easily overwhelmed. If crucial information is buried in the middle of a massive wall of text, the model frequently forgets it, hallucinates, or produces the wrong answer.

### 1.3 Claude Shannon’s Big Idea (1951)
In 1951, Claude Shannon (the father of Information Theory) proved that **human language is full of redundancy**. 

Consider this sentence:
> *"At this present moment in time, it is of the utmost importance that you take into consideration the fact that..."*

In human conversation, we use filler words, polite expressions, and grammatical glue. But the core meaning is simply:
> *"Now, consider that..."*

**The Goal of Prompt Compression:** Drop the redundant filler tokens while keeping the information-dense tokens. The LLM receives a prompt that is 3x to 10x shorter, but still answers correctly.

---

## 2. Beginner's Glossary: The Jargon Explained

Before looking at algorithms, let's demystify every technical term:

| Term | Plain English Definition | Real-World Analogy |
| :--- | :--- | :--- |
| **Token** | The basic unit of text an AI reads. It can be a word (`"apple"`), a punctuation mark (`","`), or part of a word (`"ing"`). On average, 100 words $\approx$ 130 tokens. | Syllables or building blocks of text. |
| **Language Model (LM)** | A neural network trained to predict what the next word will be, given the previous words. | A supercharged smartphone auto-complete keyboard. |
| **Causal / Autoregressive LM** | A model that reads and generates text strictly in one direction: from **left to right** (e.g., GPT, LLaMA). It cannot look into the future. | A person reading a book one word at a time through a narrow peephole moving left-to-right. |
| **Probability $P(w_i \mid w_{<i})$** | The model's confidence (from 0% to 100%) that the next word will be $w_i$, given all previous words $w_{<i}$. | If the sentence is *"The sky is..."*, $P(\text{"blue"})$ might be 85%, while $P(\text{"pizza"})$ is 0.0001%. |
| **Perplexity (PPL)** | A mathematical score of how **surprised** the model is by a word. Lower perplexity = very predictable; Higher perplexity = surprising / unexpected. | If you hear *"Once upon a..."*, you are **not surprised** by `"time"` (Low PPL). If you hear *"Once upon a... pineapple"*, you are **very surprised** (High PPL). |
| **KV Cache (Key-Value Cache)** | A temporary memory store where a Transformer saves calculations it already did for past words, so it doesn't need to re-read the sentence from the beginning every time. | Scratchpad memory so you don't have to re-do math homework steps. |
| **Compression Rate ($\tau$)** | The fraction of tokens kept. $\tau = \frac{\text{Kept Tokens}}{\text{Original Tokens}}$. If $\tau = 0.20$, you keep 20% of the words. | A 20% retention rate. |
| **Compression Ratio** | The inverse of rate: $\frac{1}{\tau}$. If $\tau = 0.20$, the ratio is $1 / 0.20 = 5\times$ compression. | Compressing a 1,000-word essay down to 200 words. |

---

## 3. How LLMLingua-1 Works: The 3 Core Components

LLMLingua-1 takes a raw prompt and compresses it through a sequential 3-step pipeline:

```
[Raw Prompt]
(Instruction + Demonstrations + Question)
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ Step 1: Budget Controller (Coarse-Grained Pruning)       │
│  - Protects sensitive parts (Instruction & Question)    │
│  - Drops entire useless demonstrations using Perplexity  │
│  - Redistributes leftover token budget                   │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: Iterative Token-Level Compression (ITPC)         │
│  - Splits text into 100-token chunks                     │
│  - Re-evaluates each token's surprise using KV-cache     │
│  - Drops individual filler words dynamically             │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Step 3: Distribution Alignment (Offline)                 │
│  - Small LM is tuned to align its "surprise" with GPT-4  │
└──────────────────────────────────────────────────────────┘
                             │
                             ▼
[Compressed Prompt] ──> Sent to Target LLM (e.g. GPT-4)
```

---

### Step 1: Budget Controller (Coarse-Grained Pruning)

#### The Problem: Not All Words Are Created Equal
A standard in-context learning prompt is composed of three distinct sections:
$$x = (x^{\text{ins}}, x^{\text{dems}}, x^{\text{que}})$$

1. **Instruction ($x^{\text{ins}}$):** *"Solve this math problem and format the final answer in square brackets."*  
   $\rightarrow$ **Extremely sensitive.** If you delete words here, the model forgets how to format the answer or misses a constraint.
2. **Question ($x^{\text{que}}$):** *"What is the total revenue of company XYZ in 2023?"*  
   $\rightarrow$ **Extremely sensitive.** If you drop `"XYZ"` or `"2023"`, the question is completely ruined.
3. **Demonstrations / Examples ($x^{\text{dems}}$):** 5 to 10 example questions and answers showing the model how to solve problems step-by-step.  
   $\rightarrow$ **High redundancy.** The model doesn't need all 10 examples. Some examples are repetitive or uninformative.

#### How the Budget Controller Solves It:
Instead of applying a flat cut (e.g. cutting 70% of everything), the Budget Controller assigns different budgets:

1. **Protect Sensitive Parts:**
   - It fixes a high retention rate for the Instruction: $\tau_{\text{ins}} = 0.85$ (keeps 85% of words).
   - It fixes a high retention rate for the Question: $\tau_{\text{que}} = 0.90$ (keeps 90% of words).

2. **Calculate the Remaining Budget for Demonstrations:**
   Suppose the user wants an overall compression rate $\tau = 0.30$ (keep 30% of the entire prompt). The controller calculates how many tokens are left for the demonstrations:
   $$\tau_{\text{dems}} = \frac{\tau \cdot L - (\tau_{\text{ins}} \cdot L_{\text{ins}} + \tau_{\text{que}} \cdot L_{\text{que}})}{L_{\text{dems}}}$$
   *(where $L$ is the total token count, and $L_{\text{ins}}, L_{\text{que}}, L_{\text{dems}}$ are the lengths of each part).*

3. **Demonstration Filtering:**
   - For each demonstration example, the small LM calculates its **average Perplexity (PPL)**.
   - High PPL means the example teaches unusual, informative reasoning. Low PPL means it's trivial boilerplate.
   - The controller ranks the examples from highest PPL to lowest PPL and **discards the lowest ones** until the demonstration token limit is met.

4. **Budget Reallocation:**
   If coarse filtering pruned an entire example and left extra room in the budget, those unused tokens are handed back to the Instruction and Question so they don't have to be cut aggressively!

---

### Step 2: Iterative Token-Level Compression (ITPC)

Now that the useless demonstration examples are dropped, we zoom in to remove **individual filler words** (fine-grained pruning).

#### The Trap: Why Naive Perplexity Fails
You might ask: *"Why can't we just calculate the perplexity of every word in the sentence once, and drop the lowest 50%?"*

Because language is **interdependent**. Consider this sentence:
> *"The quick brown fox jumps over the lazy dog."*

If you calculate probabilities once, maybe the model thinks `"quick"` and `"fox"` are redundant. But if you delete `"quick"`, the probability of the *next* word completely changes! Deleting one word changes the context for all subsequent words.

#### The ITPC Solution: Chunking + KV Cache
To fix this without re-running the entire model millions of times:

1. **Divide into Segments:** The prompt is split into sequential chunks of $S = 100$ tokens:
   $$S = \{s_1, s_2, \dots, s_m\}$$
2. **Evaluate Chunk $j$ Conditioned on Kept Tokens:**
   When evaluating tokens in chunk $s_j$, the model conditions its probability calculation **only on the tokens that survived from previous chunks**:
   $$p(s_{j,i} \mid s_{j,<i}, \tilde{s}_{<j})$$
3. **KV Cache Speedup:**
   Because a Transformer caches previous key-value states, it doesn't re-calculate past chunks from scratch. It only evaluates the 100 tokens in the current chunk against the cached surviving tokens.
4. **Dynamic Threshold ($\gamma_j$):**
   In chunk $j$, it finds a numerical threshold $\gamma_j$ such that keeping tokens with $\text{PPL} > \gamma_j$ exactly meets that segment's target budget.
5. **Surviving Tokens:** All tokens with low surprise ($\le \gamma_j$) are deleted; high surprise tokens are saved into the surviving prompt.

---

### Step 3: Distribution Alignment (The Brain Synchronization)

#### The Problem: Different Models Have Different "Surprise"
LLMLingua uses a **small LM** (like a 7B LLaMA or GPT-2) to decide which words to drop. But the **target LLM** that will actually answer the question is a giant model like **GPT-4**.

If the small model finds a word "boring/predictable" and deletes it, GPT-4 might actually need that word! Their probability distributions don't naturally match.

#### The Solution:
The authors fine-tuned the small LM on **52,000 instruction-following examples** generated by the target LLM (Stanford Alpaca dataset). This aligns the small model's understanding of language with GPT-4's, so both models agree on which words are essential.

---

## 4. Let's Build It: Step-by-Step Code Walkthrough

Here is how you can write a functional mini-version of LLMLingua from scratch in Python:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class MiniLLMLingua:
    def __init__(self, model_name="gpt2", device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"Loading base language model: {model_name} on {device}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def calculate_token_perplexities(self, input_ids):
        """
        Calculates the perplexity (surprise) of each token in the prompt.
        Surprise = -log(Probability of token given prior tokens)
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits  # Shape: (1, seq_len, vocab_size)

            # Shift logits and labels to align next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # CrossEntropy loss per token gives negative log-likelihood (NLL)
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            # The first token has no predecessor, so we prepend loss 0.0
            full_losses = torch.cat([torch.tensor([0.0], device=self.device), token_losses])
            return full_losses

    def compress_segment(self, text, retention_rate=0.5):
        """
        Compresses a single text block by keeping only the top tokens
        with the highest perplexity (highest information content).
        """
        tokens = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        losses = self.calculate_token_perplexities(tokens)
        
        num_tokens = tokens.size(1)
        k = max(1, int(num_tokens * retention_rate))  # How many tokens to keep

        # Find the threshold: get the indices of the top k highest loss tokens
        top_indices = torch.topk(losses, k=k).indices
        
        # Sort indices to preserve original reading order
        sorted_indices = torch.sort(top_indices).values

        # Reconstruct the compressed text
        kept_tokens = tokens[0, sorted_indices]
        return self.tokenizer.decode(kept_tokens, skip_special_tokens=True)

    def compress_prompt(self, instruction, context, question, target_rate=0.3):
        """
        Budget Controller in action:
        Protects Instruction & Question, compresses Context heavily!
        """
        # 1. Budget Controller allocations:
        compressed_ins = self.compress_segment(instruction, retention_rate=0.85)
        compressed_que = self.compress_segment(question, retention_rate=0.90)
        
        # Context takes the aggressive cut
        compressed_ctx = self.compress_segment(context, retention_rate=target_rate)

        # 2. Reassemble the prompt
        final_prompt = f"{compressed_ins}\n\nContext:\n{compressed_ctx}\n\nQuestion:\n{compressed_que}"
        return final_prompt


# --- Test Run ---
if __name__ == "__main__":
    compressor = MiniLLMLingua(model_name="gpt2")
    
    instruction = "Read the following document carefully and answer the question."
    context = (
        "The City Council met on Tuesday afternoon at 2:00 PM in the central municipal chamber. "
        "The members discussed the allocation of three million dollars for public road renovations "
        "and bridge reinforcement across twenty distinct regional zones."
    )
    question = "How much money was allocated for road renovations?"

    compressed = compressor.compress_prompt(instruction, context, question, target_rate=0.4)
    print("=== COMPRESSED PROMPT ===")
    print(compressed)
```

---

## 5. Strengths and Inherent Flaws of LLMLingua-1

### The Good:
- **Massive Token Savings:** Achieves **5x to 20x compression** on few-shot reasoning tasks (like math word problems in GSM8K) with almost zero loss in accuracy.
- **Structural Awareness:** Protecting the question and instructions stops the model from mangling critical constraints.

### The Inherent Flaws (Why LLMLingua-2 Was Needed):
1. **Unidirectional Peephole:** Causal models (GPT/LLaMA) read text strictly left-to-right. When looking at token #5, they have no idea what token #20 is. In human language, whether word #5 is important often depends on what comes *later*!
2. **Perplexity Is an Indirect Proxy:** Perplexity measures *"is this word predictable?"*, not *"is this word necessary for the downstream task?"*. Sometimes a predictable word is still necessary for sentence logic.
3. **Hardware Bottleneck:** Running iterative token-level compression with a 7B parameter model requires **~16.6 GB of GPU VRAM** and takes 2 to 3 seconds per prompt.

*These three limitations prompted Microsoft researchers to invent **LLMLingua-2**.*
