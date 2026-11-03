# Classification vs. Regression: What is the Output Head Doing?

**User Question:** *"Are we using linear regression? And softmax to make the predictions?"*

---

## 1. Quick Answer

* **Are we using Linear Regression?**  
  **No, it is Logistic Regression (Linear Classification).**  
  - Regression predicts a *continuous number* (e.g., predicting house prices: \$250,000, \$410,500).  
  - Classification predicts a *category or probability* (e.g., predicting **DISCARD (0)** vs. **PRESERVE (1)**).
* **Are we using Softmax to make the predictions?**  
  **YES, exactly.** Softmax converts the raw output scores from the linear layer into probabilities between 0% and 100%.

---

## 2. Where the Confusion Comes From: The "Linear Layer"

The formula looks just like linear regression:
$$z = W \cdot h + b$$

In machine learning and neural networks:
1. The operation $W \cdot h + b$ is called a **Linear Transformation** (in PyTorch: `nn.Linear(768, 2)`).
2. But what you do with $z$ determines whether it is regression or classification:

| Operation | Output Type | Name | Purpose |
| :--- | :--- | :--- | :--- |
| **Output $z$ directly** | Continuous number ($-\infty$ to $+\infty$) | **Linear Regression** | Predicting prices, temperatures, stock values. |
| **Pass $z$ into Softmax** | Probabilities ($0.0$ to $1.0$, summing to $1$) | **Logistic Regression / Softmax Classification** | Predicting classes (**DISCARD** vs. **PRESERVE**). |

---

## 3. Step-by-Step Flow: From Vector to Probability

Here is what happens inside the model for every single token:

```
[Context Vector h_i] 
(768 numbers from the BERT encoder)
         │
         ▼
[Linear Layer: W * h_i + b]   <─── Linear Transformation
         │
         ▼
[Raw Logits: z_i]
(e.g., [-1.2, +3.4])
         │
         ▼
     [Softmax]                 <─── Softmax Activation
         │
         ▼
[Probabilities]
P(discard) = 0.01
P(preserve) = 0.99
```

### The Math with Real Numbers:
Suppose for the word `"intelligence"`, the linear head produces two raw numbers (called **logits**):
$$z = [z_{\text{discard}} = -1.2, \quad z_{\text{preserve}} = +3.4]$$

Now, **Softmax** turns those two raw numbers into valid probabilities:
1. $e^{-1.2} = 0.301$
2. $e^{3.4} = 29.964$
3. Sum $= 0.301 + 29.964 = 30.265$

Now divide each:
- **Probability of Discard:** $\frac{0.301}{30.265} = \mathbf{0.010} \implies \mathbf{1.0\%}$
- **Probability of Preserve:** $\frac{29.964}{30.265} = \mathbf{0.990} \implies \mathbf{99.0\%}$

Because $P(\text{preserve}) = 99.0\%$, the word is given high priority and kept in the compressed prompt.

---

## 4. Summary Table

| Component | What it is Called | What it Does |
| :--- | :--- | :--- |
| **$W \cdot h_i + b$** | **Linear Layer (Fully Connected Layer)** | Combines the 768 context features into 2 raw decision scores. |
| **Softmax** | **Probability Activation Function** | Squashes the 2 scores into percentages summing to 100%. |
| **Whole System** | **Token Classification (Logistic Regression)** | Classifies each word as `0 (Discard)` or `1 (Preserve)`. |
