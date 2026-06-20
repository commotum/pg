# V1

## Type and Value Embeddings

Standard token-based approaches in large language models (LLMs) typically rely on constrained vocabularies due to computational and representational limitations. Common data types, such as RGB colors or 64-bit integers (int64), present an immense combinatorial challenge. For instance, the RGB color space alone contains over 16 million unique values, vastly exceeding the vocabularies used by popular models like LLaMA-3.2 (\~128K tokens), DeepSeek-V3 (\~129K tokens), Claude (\~65K tokens), GPT-4 (\~100K tokens), GPT-4o (\~200K tokens), and Gemma (\~256K tokens). Likewise, the int64 data type spans approximately $9.22 \times 10^{18}$ distinct values, rendering explicit tokenization computationally infeasible.

To address this challenge, we propose representing tokens through both a **type** and a **value**, enabling a structured embedding approach that significantly reduces model complexity. Under this strategy, each data type is assigned a compact, low-dimensional representation systematically projected into a higher-dimensional embedding space through a learned linear transformation—an operation we term an *up-projection*.

Taking RGB values as a concrete example, each RGB token can be efficiently modeled as a purely imaginary quaternion, mapping the hexadecimal range `[00, FF]` to the floating-point range `[-1.0, 1.0]` using BF16 precision. Consequently, each RGB token is succinctly represented by just four BF16 values. To achieve a high-dimensional token embedding (e.g., 512 dimensions), this representation requires only a learned real-valued matrix of shape $512 \times 4$. Remarkably, this reduces the embedding parameters required for all possible RGB tokens—from explicitly storing embeddings for more than 16 million distinct values—to merely 2,048 parameters.

Further efficiency can be attained by employing quaternion-valued weight matrices, effectively quartering parameter counts. While traditionally projecting four real dimensions would require 16 real weights, quaternion arithmetic allows the same projection using just four quaternion weights. Thus, the original real-valued matrix of size $512 \times 4 = 2048$ parameters becomes just 128 quaternion weights (512 real-valued parameters), substantially expanding representational efficiency without increasing complexity.

Additionally, retrieving the original RGB values from the high-dimensional embeddings is computationally straightforward due to the constant-time complexity $O(1)$ of quaternion inversion. By applying the inverse quaternion transformations, the original RGB values are recovered precisely from the token embeddings, providing an efficient and exact decoding mechanism suitable for real-world applications.

Quaternion algebra is abundantly covered in existing literature, and the cited references provide thorough treatment; thus, we shall not beleaguer you here with their basic properties.


Quaternions came from Hamilton after his really good work had been done, and though beautifully ingenious, have been an unmixed evil to those who have touched them in any way.
- Lord Kelvin



################################################################################



# V2

## Type & Value Embeddings

### Scalable Tokenization for Large Value Spaces


#### **Problem.**  

Next‑token prediction in LLMs typically computes logits by comparing the final hidden state against **every row** of a learned embedding table, followed by a (|\mathcal V|)-way softmax. This is tractable for vocabularies (\sim 10^5), but intractable for tokens whose **value spaces are huge**, e.g., RGB ((256^3=16{,}777{,}216)) or 64‑bit integers ((\approx 9.22\times 10^{18})). A row‑per‑value table explodes parameters and compute.


#### **Proposed solution.**  

Represent each token as **(TYPE, VALUE)** and replace the row‑per‑value table with: (i) a **minimal representation** (MR) for the value (a small, structured vector), and (ii) an **embedding expansion**—a learned linear projection that lifts the MR to the model dimension. At the output, avoid a (|\mathcal V|)-softmax by decoding the value directly from the final hidden state using a lightweight, structured reader.


#### **Concrete example (real weights).**  

For RGB, use an MR (q\in\mathbb R^4) (e.g., (q=[0,r,g,b])). A real expansion applies a single matrix (W_{\text{real}}\in\mathbb R^{d_{\text{model}}\times 4}):
[
Y ;=; W_{\text{real}},q ;\in; \mathbb R^{d_{\text{model}}}.
]
This needs only (4d_{\text{model}}) parameters (e.g., (2{,}048) for (d_{\text{model}}=512)), independent of the (16.7)M RGB values.


#### **Decoding challenge with real expansion.**  

To recover (q) from a generic final hidden state (h_T), one must either (a) compute/learn a **left inverse** of the expansion (often unstable after nonlinear mixing), or (b) add a **learned contraction** (C\in\mathbb R^{4\times d_{\text{model}}}) to map back to 4‑D—effectively **doubling** head parameters and latency.

#### **Quaternion expansion (our fix).**  

Replace the single real matrix with a **bank of quaternion weights** applied blockwise. Partition the model dimension into (N_q=d_{\text{model}}/4) 4‑D blocks and learn one quaternion (W_i\in\mathbb R^4) per block (per TYPE). The expansion is
[
y_i ;=; q \otimes W_i \in \mathbb R^4,\qquad
Y ;=; \mathrm{vec}(y_1,\ldots,y_{N_q}).
]
A real (4!\times!4) block uses 16 parameters; a quaternion uses 4, giving an immediate **(4\times)** reduction. Crucially, quaternions admit an **(O(1))** inverse, (W_i^{-1}=\mathrm{conj}(W_i)/|W_i|^2), so we **do not** need a learned contraction—yielding an effective **(\approx 8\times)** reduction versus “real expansion + contraction.”


#### **Decoding (voting Gaussian, no softmax).**  

We read the value directly from (h_T) with a blockwise “vote” and fuse:

* **Split into clues.**  Reshape (h_T) into (\widehat{y}*0,\ldots,\widehat{y}*{N_q-1}\in\mathbb R^4).
* **Per‑block vote.**  Invert each block with its weight:
  [
  \widehat{q}_i ;=; \widehat{y}_i \otimes W_i^{-1}
  ;=; \widehat{y}_i \otimes \frac{\mathrm{conj}(W_i)}{|W_i|^2}.
  ]
* **Fuse votes (LS/ML mean).**  Treating votes as noisy measurements, the least‑squares estimate is
  [
  \boxed{;\mu_q ;=;
  \frac{\sum_i \widehat{y}_i \otimes \mathrm{conj}(W_i)}
  {\sum_i |W_i|^2};}
  ]
  with a scalar **spread**
  (
  s^2=\frac{\sum_i |W_i|^2,|\widehat{q}_i-\mu_q|^2}{\sum_i |W_i|^2}
  )
  as confidence (skinny bell curve when votes agree; wide when they don’t).
* **Discretize locally.**  Around (\mu_q), evaluate a tiny neighborhood of RGB bins (e.g., 7 per channel → (\le 343) candidates), score by reconstruction error (\sum_i|\widehat{y}_i - q(c)\otimes W_i|^2), and return top‑(k). No (|\mathcal V|)-softmax anywhere.


#### **Why this still works under nonlinear attention/MLPs.**  

We are **not** inverting the transformer. The model is trained **end‑to‑end** so that, after all nonlinear mixing, (h_T) **writes the right clues into the right 4‑D slots** for our decoder. This mirrors why a standard GPT uses a **linear** output head after many nonlinear layers: the representation becomes **decodable** because the loss demands it. Here, each 4‑D block in (h_T) is a **noisy measurement** under its quaternion code; we invert per block to get votes and fuse them (least‑squares/“Gaussian” mean) to obtain a best guess and confidence. Nonlinearities change how information is **encoded**, not whether it is **recoverable** by the designed head; the value loss pushes the network to place recoverable signal in those blocks.


#### **bf16‑friendly RGB↔quaternion mapping (MR).**  

Map 8‑bit channels ((r,g,b)\in{0,\ldots,255}) to a centered, power‑of‑two grid (purely imaginary quaternion):
[
\tilde r=\tfrac{2r-255}{256},\quad
\tilde g=\tfrac{2g-255}{256},\quad
\tilde b=\tfrac{2b-255}{256},\qquad
Q ;=; 0+\tilde r,\mathbf i+\tilde g,\mathbf j+\tilde b,\mathbf k.
]
The step (2/256=1/128) aligns with bf16 spacing, keeping adjacent bins distinct. Decode (\widehat Q=[0,\hat r,\hat g,\hat b]) via
(
\widehat r_{\text{8-bit}}=\mathrm{round}!\big((256\hat r+255)/2\big)
)
(and analogously for (g,b)), clamped to ([0,255]).

#### **Training.**  

Use a small TYPE classifier (or the same mechanism on a TYPE segment), plus a VALUE loss on (\mu_q) (L2 or Gaussian NLL). An optional **vote‑tightening** penalty (\sum_i |W_i|^2|\widehat{q}_i-\mu_q|^2) encourages confident estimates. Parameterize (W_i=s_i,\hat W_i) with (|\hat W_i|=1), (s_i>0) (store (\log s_i)) for stable inversion. All operations are differentiable and implemented with standard tensor ops.


#### **Summary.**  

By replacing a row‑per‑value table with a type‑conditioned **quaternion expansion** and a **voting** decoder, we obtain (O(d_{\text{model}})) inference cost, **(\approx 8\times)** head‑parameter savings versus real expansion + contraction, calibrated predictions (mean + confidence), and **no** (|\mathcal V|)-softmax—even for value spaces with tens of millions of elements.


################################################################################


# V3

## Type & Value Embeddings

### Scalable tokenization for large value spaces

#### **Problem**

Most LLMs treat “a token” as “a row in a giant table.” That works when the vocabulary is (\sim 10^5) items, but it collapses for tokens whose **value spaces are enormous**—think RGB colors ((256^3 = 16{,}777{,}216)) or 64‑bit integers ((\approx 9.22\times 10^{18})). A row‑per‑value table explodes parameters and makes the usual (|\mathcal V|)-way softmax impractical.

#### **Proposed solution**

Split every token into **TYPE** and **VALUE**. Encode the value with a **minimal representation** (MR)—a tiny, structured vector—then **expand** it into the model dimension with a learned linear map (an *up‑projection*). On the way out, **do not** run a (|\mathcal V|)-softmax; instead, read the value back directly from the final hidden state with a lightweight, structured decoder.

---

### Concrete example (RGB with a real expansion)

For RGB, use a 4‑D MR (q) that acts like a purely imaginary quaternion:
(q=[0,\tilde r,\tilde g,\tilde b]), one coordinate per channel. A single real matrix
(W_{\text{real}}\in\mathbb R^{d_{\text{model}}\times 4}) expands it:
[
Y = W_{\text{real}},q \in \mathbb R^{d_{\text{model}}}.
]
This needs only (4d_{\text{model}}) parameters—e.g., **2,048** when (d_{\text{model}}=512)—**independent** of the 16.7M RGB values.

##### Where the real expansion gets clumsy

Recovering (q) from a general final state (h_T) either demands a learned left‑inverse (touchy after nonlinear mixing) or an **extra contraction** (C\in\mathbb R^{4\times d_{\text{model}}}) to pull (h_T) back to 4‑D—doubling head parameters and adding latency.

---

### Quaternion expansion (the fix)

Replace the single big real matrix with a **bank of quaternion weights** applied blockwise. Partition the model dimension into (N_q=d_{\text{model}}/4) contiguous 4‑D blocks and learn one quaternion weight (W_i) per block (optionally per TYPE). For block (i):
[
y_i = q \otimes W_i \in \mathbb R^4,\qquad
Y=\mathrm{vec}(y_1,\ldots,y_{N_q}).
]

Why this helps:

* **4× fewer parameters (immediately).** A dense real (4\times 4) block uses 16 real weights; one quaternion uses **4**. Across blocks, that’s a clean **4×** reduction.
* **No separate contraction.** Quaternions have an **(O(1))** inverse, (W_i^{-1}=\mathrm{conj}(W_i)/|W_i|^2). We can decode value blocks directly, so there’s no extra (C). Against “real expansion + contraction,” the quaternion head yields an effective **(\approx 8\times)** savings.
* **Exact, stable inversion.** The conjugate‑over‑norm form makes blockwise inversion well‑behaved.

---

### Decoding (no (|\mathcal V|)-softmax)

We read values directly from (h_T) with a simple “blockwise votes + fuse” routine:

1. **Split into clues.** Reshape (h_T) into (\widehat y_0,\ldots,\widehat y_{N_q-1}\in\mathbb R^4).
2. **Per‑block vote.** Invert each block with its weight:
   [
   \widehat q_i=\widehat y_i\otimes W_i^{-1}
   =\widehat y_i\otimes\frac{\mathrm{conj}(W_i)}{|W_i|^2}.
   ]
3. **Fuse votes (least‑squares / ML mean).** Treat votes as noisy measurements and take the weighted mean:
   [
   \boxed{\ \mu_q=\frac{\sum_i \widehat y_i\otimes \mathrm{conj}(W_i)}
   {\sum_i |W_i|^2}\ }.
   ]
   A scalar spread
   (
   s^2=\frac{\sum_i |W_i|^2,\lVert \widehat q_i-\mu_q\rVert^2}{\sum_i |W_i|^2}
   )
   serves as a **confidence** (tight when blocks agree; wide when they don’t).
4. **Local discretization.** Around (\mu_q), score a tiny neighborhood of RGB bins (e.g., 7 per channel → (\le 343) candidates) by reconstruction error
   (\sum_i\lVert\widehat y_i - q(c)\otimes W_i\rVert^2),
   and return the top‑(k). No global softmax anywhere.

---

### Why this survives nonlinear attention/MLPs

We’re **not** inverting the entire transformer. Training is end‑to‑end, so the network **learns** to place recoverable signal into the designated 4‑D slots. This is the same reason a standard GPT can end with a **linear** output head after many nonlinear layers: the loss shapes the representation. Here, each 4‑D block becomes a “noisy clue”; the quaternion code lets us invert per block and fuse the votes cleanly.

---

### bf16‑friendly RGB ↔ quaternion mapping (MR)

Map 8‑bit channels ((r,g,b)\in{0,\ldots,255}) onto a centered, power‑of‑two grid so adjacent bins remain distinct under **bf16**:

[
\tilde r=\frac{2r-255}{256},\quad
\tilde g=\frac{2g-255}{256},\quad
\tilde b=\frac{2b-255}{256},\qquad
q=[0,\tilde r,\tilde g,\tilde b].
]

Decode (\widehat q=[0,\hat r,\hat g,\hat b]) via
(\widehat r_{\text{8-bit}}=\mathrm{round}!\big((256\hat r+255)/2\big)) (and analogously for (g,b)), then clamp to ([0,255]).

---

### Training

Use a small TYPE classifier (or reserve a TYPE segment in (h_T)), plus a VALUE loss on (\mu_q) (L2 or Gaussian NLL). An optional **vote‑tightening** penalty
(\sum_i |W_i|^2\lVert \widehat q_i-\mu_q\rVert^2)
encourages confident, consistent blocks. Parameterize (W_i) as (W_i=s_i\hat W_i) with (|\hat W_i|=1) and (s_i>0) (store (\log s_i)) for stable inversion and training.

---

### Summary

By replacing a row‑per‑value table with a **type‑conditioned quaternion expansion** and a **voting decoder**, we keep inference cost (O(d_{\text{model}})), avoid any (|\mathcal V|)-softmax, recover values directly with a calibrated mean‑and‑spread, and cut head parameters by **(\approx 8\times)** versus a real expansion plus a learned contraction—even when the value space runs to tens of millions.

> *Quaternion algebra is well covered in the literature, so we won’t recap basics here.*
> “Quaternions came from Hamilton after his really good work had been done, and though beautifully ingenious, have been an unmixed evil to those who have touched them in any way.” — *Lord Kelvin*

---

IMPROVEMENTS:

"Each value is first mapped to a minimal representation, then lifted into the model dimension by a learned up-projection (a structured embedding encoder)."