# Building LLMLingua-2 from Scratch: The Complete Beginner's Guide

> **Paper Title:** *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression* (Findings of ACL 2024)  
> **Core Concept:** Supervised Token Classification via Bidirectional Transformer Encoders  
> **Target Audience:** Anyone starting from zero who wants to understand and build LLMLingua-2 step-by-step.

---

## 1. The Paradigm Shift: Why Scrap LLMLingua-1?

In LLMLingua-1, we used a causal language model (like LLaMA or GPT-2) and calculated **Perplexity (Surprise)** to decide which words to drop. 

While clever, it had three major flaws:
1. **Perplexity Is the Wrong Metric:** A language model is trained to predict the *next word*. It was never trained to decide *which words are necessary for compression*. Just because a word is predictable does not mean you can safely delete it.
2. **The "Unidirectional Blindspot":** Causal models only look **left-to-right**. But in human language, the importance of a word frequently depends on what words appear **after** it.
3. **Too Heavy & Slow:** Running a 7B model iteratively with KV-cache takes ~16.6 GB of VRAM and ~2.5 seconds per prompt.

### The Breakthrough Idea of LLMLingua-2
Instead of guessing word importance using perplexity, **treat prompt compression as a Binary Token Classification problem**:
- Look at every word in the text.
- Use a **Bidirectional Transformer Encoder** (like XLM-RoBERTa or BERT) that reads both left and right context simultaneously.
- For each word, classify: **$1$ (Preserve)** or **$0$ (Discard)**.

---

## 2. Beginner's Glossary: The Jargon Explained

| Term | Plain English Definition | Real-World Analogy |
| :--- | :--- | :--- |
| **Token Classification** | Giving an individual tag/label to every single token in a text (e.g., Tagging each word as Keep or Delete). | A teacher going through an essay with a yellow highlighter to mark only the essential words. |
| **Transformer Encoder** | A neural network (like BERT or RoBERTa) that reads the entire sentence in all directions at once (**bidirectional**). | Reading an entire sentence as a complete picture, rather than reading word-by-word through a peephole. |
| **Data Distillation** | Using a huge, smart model (GPT-4) to create clean training examples, and using those examples to train a tiny model to do the exact same job. | A master chef teaching an apprentice their best recipes. |
| **Lemmatization** | Reducing different grammatical forms of a word back to its base root dictionary form (lemma). e.g., `"running"`, `"ran"`, `"runs"` all become `"run"`. | Stripping the paint off a table to see the bare wood underneath. |
| **Sliding Window** | A technique where you restrict your search to a small local neighborhood around your current position, rather than scanning the entire document. | Looking for your lost keys only in the living room where you were sitting, instead of searching the entire city. |
| **Softmax** | A mathematical function that converts raw neural network numbers into valid percentages/probabilities that add up to 100% (1.0). | Converting test points into a percentage grade. |
| **Subword Pooling** | Averaging the predicted probabilities of subwords (e.g., `["un", "##known"]`) to produce a single score for the whole word (`"unknown"`). | Averaging exam scores across multiple questions to get a single student grade. |

---

## 3. The Complete 5-Step Pipeline of LLMLingua-2

```
MeetingBank Transcripts (5,169 real-world meeting records)
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: Data Distillation from GPT-4                    │
│  - 5 strict extraction rules (delete only, no adding)   │
│  - Small chunks (<= 512 tokens ending with a period)    │
│  - Natural dynamic compression without forced ratios    │
└────────────────────────┬────────────────────────────────┘
                         │ (Raw compressed text string)
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Data Annotation (Algorithm 1)                   │
│  - Maps GPT-4's text back to original text words        │
│  - Uses Sliding Window + Bidirectional Search           │
│  - Uses spaCy lemmatization for fuzzy word matching     │
└────────────────────────┬────────────────────────────────┘
                         │ (Assigned 0 / 1 labels per word)
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Quality Control & Filtering                     │
│  - Variation Rate (VR): Purges hallucinations (top 5%)  │
│  - Alignment Gap (AG): Purges mapping errors (top 10%)  │
└────────────────────────┬────────────────────────────────┘
                         │ (Clean Gold Training Data)
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: Model Training                                  │
│  - Train XLM-RoBERTa (355M) or mBERT (110M)             │
│  - Learns P(preserve) vs P(discard) with Cross-Entropy  │
└────────────────────────┬────────────────────────────────┘
                         │ (Trained Small Compressor Model)
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 5: Test-Time Inference                             │
│  - Single bidirectional forward pass (takes ~0.4s)      │
│  - Select top-N words with highest P(preserve)          │
│  - Reconstruct prompt in original sequence order        │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Step-by-Step Deep Dive

### Step 1: Data Distillation from GPT-4
To train a model to classify words as Keep vs. Discard, we need thousands of examples of original texts paired with their compressed versions.

The authors picked **MeetingBank** (transcripts of city council meetings). Why? Because spoken meetings are full of banter, filler words, stuttering, and repetitive phrasing — the perfect testbed for compression!

#### The Prompt: The 5 Golden Rules
When you ask GPT-4 to summarize, it loves to rewrite, paraphrase, and invent words. To force GPT-4 to act strictly as an **extractive compressor**, they used this strict prompt:
1. You can **ONLY** remove unimportant words.
2. Do **NOT** reorder the original words.
3. Do **NOT** change the original words.
4. Do **NOT** use abbreviations or emojis.
5. Do **NOT** add new words or symbols.

#### The Chunking Trick
When GPT-4 is given a very long text (over 1,000 tokens), it gets lazy and deletes up to 90% of the content, dropping vital facts.  
To prevent this, every meeting transcript is split into **chunks of $\le 512$ tokens** ending at a complete sentence period. GPT-4 compresses each small chunk individually.

---

### Step 2: Data Annotation (Algorithm 1)

#### The Problem:
GPT-4 outputs a **string of text**, such as:
> `Item 15 City Manager Recommendation adopt three resolutions.`

It does **not** give us the boolean label list:
> `[1, 1, 1, 1, 1, 1, 1...]`

We must align every word in GPT-4's compressed text back to the original text. But GPT-4 is imperfect and causes 3 headaches:
1. **Ambiguity:** A word like `"program"` might appear 6 times in the original paragraph. Which one did GPT-4 keep?
2. **Grammar Variation:** Even with instructions, GPT-4 sometimes outputs `"adopt"` when the original was `"adopted"`, or `"property"` when the original was `"properties"`.
3. **Reordering:** GPT-4 occasionally flips two adjacent words.

---

#### Algorithm 1: How the Alignment Works

Here is the exact algorithm from the paper, explained with a visual pointer:

```
Original Text List (S_ori):
Index:  0     1     2       3     4       5        6               7
Word:  [Item, 15,   report, from, City,   Manager, Recommendation, adopt]

Compressed Word List (S_comp):
Word:  [Item, 15,   City,   Manager, Recommendation, adopt]
```

#### Step-by-Step Execution:
1. Initialize all labels in $S_{\text{ori}}$ to `False` (`0`).
2. Set a pointer `prev = 0` (this tracks where we found our last match in the original text).
3. Set a sliding window radius (e.g. $s = 20$).
4. For each word $w$ in the compressed list $S_{\text{comp}}$:
   - **Search Forward:** Look at positions `prev + 1, prev + 2, ... prev + s/2`.
   - **Search Backward:** Look at positions `prev - 1, prev - 2, ... prev - s/2`.
   - At each candidate position, check **Fuzzy Matching** using spaCy lemmatization:
     $$\text{Is lemma}(w) == \text{lemma}(S_{\text{ori}}[\text{candidate}])?$$
   - **If a match is found:**
     - Set $L[\text{candidate}] = \text{True}$.
     - Update `prev = candidate` (the window moves forward!).
     - Stop searching and proceed to the next compressed word.

#### Why This Works:
- **Sliding Window:** Because it only searches near `prev`, if `"program"` appears at index 5 and index 200, it matches index 5 first. It never jumps wildly across the text.
- **Bidirectional Search:** Looking backward (`prev - i`) handles cases where GPT-4 accidentally swapped two words.
- **spaCy Lemmatization:** `"consenting"` and `"consent"` share the root `"consent"`, so they match cleanly!

---

### Step 3: Quality Control & Filtering

Even with Algorithm 1, some distilled examples are flawed. The authors filter them out using two metrics:

#### 1. Variation Rate (VR) — Hallucination Filter
Measures what percentage of words in GPT-4's compressed text did not exist anywhere in the original text:
$$VR = \frac{1}{|S_{\text{comp}}|} \sum_{w \in S_{\text{comp}}} \mathbb{I}(w \notin S_{\text{ori}})$$

- If GPT-4 hallucinated a new word (e.g., added `"summary"` or an abbreviation), that word will be missing from $S_{\text{ori}}$.
- **Filter:** Discard the **top 5% worst samples** with the highest VR.

#### 2. Alignment Gap (AG) — Mapping Failure Filter
Measures the gap between words that exist vs. words Algorithm 1 actually managed to match:
- **Matching Rate (MR):** $\frac{\text{Number of original words labeled True}}{\text{Total words in original}}$
- **Hitting Rate (HR):** $\frac{\text{Number of compressed words that exist in original}}{\text{Total words in original}}$
- **Alignment Gap:**
  $$AG = HR - MR$$

- **Intuition:** In a perfect alignment, every word in $S_{\text{comp}}$ that exists in $S_{\text{ori}}$ gets matched ($HR = MR$), so $AG = 0$. If $AG$ is high, Algorithm 1 got lost or confused.
- **Filter:** Discard the **top 10% worst samples** with the highest Alignment Gap.

---

### Step 4: Model Training

Now we have a clean dataset of pairs:
$$(\text{Prompt Text } x, \text{ Gold Labels } y \in \{0, 1\}^N)$$

#### The Architecture:
We use a pretrained **XLM-RoBERTa-large** (355 million parameters) or **mBERT** (110 million parameters) and attach a linear classification layer on top:

1. **Feature Extraction:**
   Pass the text $x$ through the Transformer encoder to get a contextual embedding vector $h_i$ for each token:
   $$h = f_\theta(x)$$
2. **Binary Classification Head:**
   Project each vector $h_i$ into 2 logits (scores for [Discard, Preserve]) using a learned matrix $W$ and bias $b$:
   $$p(x_i) = \text{Softmax}(W \cdot h_i + b)$$
3. **Subword Pooling:**
   If a word like `"infrastructure"` was broken by the tokenizer into `["infra", "##structure"]`, we compute the word's probability by **averaging the probabilities of its subwords**.
4. **Loss Function:**
   Standard Binary Cross-Entropy Loss:
   $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \text{CrossEntropy}(y_i, p(x_i))$$
   Trained for 10 epochs using the Adam optimizer with learning rate $1\times 10^{-5}$.

---

### Step 5: Test-Time Inference (Fast Compression)

When you run LLMLingua-2 on your machine, here is what happens under the hood:

Suppose you give it a prompt with $N = 100$ words and request **3x compression** ($\tau = 0.33$):

1. **Calculate Budget:**
   $$\tilde{N} = \tau \times N = 0.33 \times 100 = 33 \text{ words to keep}$$
2. **Single Forward Pass:**
   The entire text is fed through XLM-RoBERTa in **one single pass**. Every word gets a score $p_i = P(\text{preserve})$.
3. **Top-$\tilde{N}$ Selection:**
   Rank all 100 words by their $p_i$ score and pick the 33 words with the highest scores.
4. **Restore Original Order:**
   Sort the 33 chosen words by their original position in the text.
5. **Output:** Return the compressed string.

---

## 5. Let's Build It: Python Implementation from Scratch

### Part 1: Algorithm 1 (Data Annotation with spaCy)

```python
import spacy

# Load spaCy English pipeline for lemmatization
nlp = spacy.load("en_core_web_sm")

def annotate_labels_algorithm_1(original_text, compressed_text, window_size=20):
    """
    Implements Algorithm 1 from LLMLingua-2:
    Aligns words from compressed_text back to original_text using
    a sliding window, bidirectional search, and spaCy lemmatization.
    """
    doc_ori = nlp(original_text)
    doc_comp = nlp(compressed_text)

    # Extract words and their root lemmas
    ori_words = [token.text for token in doc_ori]
    ori_lemmas = [token.lemma_.lower() for token in doc_ori]

    comp_words = [token.text for token in doc_comp]
    comp_lemmas = [token.lemma_.lower() for token in doc_comp]

    # Initialize all labels to False (0 = discard)
    labels = [False] * len(ori_words)
    prev = 0  # Pointer to last matched position
    half_win = window_size // 2

    for c_idx, c_lemma in enumerate(comp_lemmas):
        matched = False

        # Search within local window: check right (ahead) and left (behind)
        for i in range(1, half_win + 1):
            # 1. Search right (ahead)
            right = min(len(ori_words) - 1, prev + i)
            if not labels[right] and ori_lemmas[right] == c_lemma:
                labels[right] = True
                prev = right
                matched = True
                break

            # 2. Search left (behind, handles reordering)
            left = max(0, prev - i)
            if not labels[left] and ori_lemmas[left] == c_lemma:
                labels[left] = True
                matched = True
                break

        # Fallback: if not found within window, check prev directly
        if not matched and not labels[prev] and ori_lemmas[prev] == c_lemma:
            labels[prev] = True

    return ori_words, labels


# --- Test Algorithm 1 ---
if __name__ == "__main__":
    orig = "Item 15, report from City Manager Recommendation to adopt three resolutions."
    comp = "Item 15 City Manager Recommendation adopt three resolutions."
    
    words, tags = annotate_labels_algorithm_1(orig, comp)
    print("Alignment Result:")
    for w, t in zip(words, tags):
        status = "KEEP (1)" if t else "DROP (0)"
        print(f"{w:15} -> {status}")
```

---

### Part 2: Building the Compressor Classifier with PyTorch

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class LLMLingua2Compressor(nn.Module):
    """
    XLM-RoBERTa / BERT Encoder with a Linear Token Classification Head
    """
    def __init__(self, model_name="xlm-roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Binary classification head: [P(discard), P(preserve)]
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        logits = self.classifier(sequence_output)    # (batch, seq_len, 2)
        probabilities = torch.softmax(logits, dim=-1) # (batch, seq_len, 2)
        return probabilities
```

---

### Part 3: Test-Time Inference Function

```python
def compress_prompt_inference(model, tokenizer, prompt_text, target_rate=0.4, device="cpu"):
    """
    Compresses a prompt using a single forward pass:
    1. Feeds prompt to encoder
    2. Takes P(preserve)
    3. Retains top tokens maintaining original sequence order
    """
    model.eval()
    tokens = tokenizer(prompt_text, return_offsets_mapping=True, return_tensors="pt").to(device)
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask

    with torch.no_grad():
        probs = model(input_ids, attention_mask)
        # Probability of class 1 (preserve)
        preserve_probs = probs[0, :, 1]

    # Calculate token budget
    seq_len = input_ids.size(1)
    k = max(1, int(seq_len * target_rate))

    # Pick top k indices with highest preservation probabilities
    top_indices = torch.topk(preserve_probs, k=k).indices
    sorted_indices = torch.sort(top_indices).values

    # Decode preserved tokens back to text
    kept_input_ids = input_ids[0, sorted_indices]
    compressed_text = tokenizer.decode(kept_input_ids, skip_special_tokens=True)
    return compressed_text
```

---

## 6. Comparison: LLMLingua-1 vs. LLMLingua-2

| Dimension | LLMLingua-1 (EMNLP 2023) | LLMLingua-2 (ACL 2024) |
| :--- | :--- | :--- |
| **Guiding Metric** | Perplexity (Unsupervised surprise) | Supervised $P(\text{preserve})$ probability |
| **Model Type** | Causal LM (LLaMA-7B / GPT-2) | Bidirectional Encoder (XLM-RoBERTa / mBERT) |
| **Vision Field** | Left-to-right only (unidirectional) | Full sentence simultaneously (bidirectional) |
| **Peak GPU VRAM** | **~16.6 GB** | **~1.5 GB – 2.1 GB** (8x lighter) |
| **Compression Speed** | ~2.5 seconds | **~0.4 seconds** (3x - 6x faster) |
| **Inference Step** | Iterative KV-cache chunk scoring | **Single forward pass** + Top-N selection |
| **Hallucination Risk**| Zero (Extractive) | Zero (Extractive) |
| **Hardware Requirement**| Needs large GPU or runs slow on CPU | Runs easily on your **4 GB MX150 GPU** or **Intel i7 CPU** |
