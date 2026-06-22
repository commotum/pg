# Learned Token Types as a Parameter Golf Route

## Core idea

The useful version of the type/value embedding idea is probably not a hand-built ontology like "people, places, things" from the start. For Parameter Golf, the more practical route is to discover token classes from the tokenizer, data distribution, and model weights, then use those classes as a compact prediction structure.

The goal is not just to reduce parameter count. The goal is lower bits per byte under the 16 MB artifact cap and fixed training-time constraints. A type system is only useful if it improves the compression/training tradeoff:

```text
context -> predict coarse token type/class -> predict token or value within that class
```

If this works, the model avoids treating the vocabulary as one flat set of unrelated IDs. It can spend capacity on a smaller class decision plus a more local within-class decision.

## Why hand labels are probably the wrong first step

The 1024 vocab is mostly byte fallback, punctuation, single letters, common word starts, and continuation fragments. It has almost no clean semantic entities. The 4096 vocab starts to contain names, places, dates, web/code pieces, and common words, but many tokens are still fragments such as suffixes, prefixes, and partial capitalized words.

That means a human ontology is brittle:

- A token like `John` can be a name, but `Al` is both a name and a prefix.
- A token like `New` can be geographic in `New York`, but ordinary in other contexts.
- Continuations like `tion`, `ing`, `ment`, and `able` are morphological, not semantic.
- BPE IDs are optimized for compression frequency, not clean linguistic categories.

So the better starting point is to learn useful classes from evidence rather than assign permanent semantic meanings by hand.

## Three levels of type discovery

### 1. Surface-form classes

These are cheap, deterministic classes from the token string itself:

- special/control tokens
- byte fallback tokens
- punctuation and symbols
- digits and numeric fragments
- single letters
- space-prefixed word starts
- non-space continuations
- capitalized word starts
- lowercase word starts
- prefix-like pieces
- suffix-like pieces
- URL/code-ish pieces

This is useful as a baseline and as extra features, but it will miss distributional structure.

### 2. Distributional classes

These classes are learned from token behavior in context.

Possible inputs for clustering:

- input embedding row for each token
- output head row for each token
- average hidden state before each token on a corpus sample
- left/right neighboring token statistics
- token frequency
- token length and surface flags
- starts-with-space flag
- alpha/digit/punctuation/case flags

Then cluster tokens into a moderate number of classes, for example 64, 128, or 256. The clusters do not need human-readable names. A good cluster is one that improves the prediction factorization.

This can discover classes like:

- function words
- pronouns
- suffixes
- punctuation
- name-like capitalized words
- place-like words
- dates/months/weekdays
- web/code tokens
- common nouns
- verb fragments
- rare continuation fragments

### 3. Jointly learned classes

The most ambitious version is to make the type assignment itself learned during training. Each token has a learned type/code assignment or soft membership over types, and the model learns to predict through that structure.

This is potentially powerful, but it is riskier:

- more moving parts
- more training instability
- harder checkpoint/package accounting
- harder comparison against dense baselines

For this repo, jointly learned types should come after a simpler clustered-type prototype shows promise.

## Practical experiment path

### Phase A: Create discovered type labels offline

Start with an existing baseline model and a tokenizer vocab.

1. Export token metadata for the target vocab.
2. Extract input embedding rows and output head rows from a trained or partially trained baseline.
3. Add surface features per token:
   - token length
   - starts with `▁`
   - is byte fallback
   - is punctuation
   - has digits
   - capitalization pattern
   - alpha/digit/symbol mix
4. Normalize features.
5. Cluster tokens into `K` classes, starting with `K in {64, 128, 256}`.
6. Save a mapping:

```text
token_id -> type_id
```

This gives a fixed discovered type system without changing tokenizer training.

### Phase B: Use classes as analysis first

Before changing training, inspect the clusters:

- Are special/byte/punctuation tokens separated cleanly?
- Do function words cluster together?
- Do suffixes/continuations cluster together?
- Do high-frequency word starts form stable groups?
- Are classes wildly imbalanced?
- Does increasing `K` create useful refinement or just noise?

The classes do not need perfect human semantics, but they should not look random.

### Phase C: Test a clustered output head

The most relevant BPB experiment is output-side, because larger vocabs slow down partly because the model must score the full vocabulary every step.

A clustered output head would factor prediction:

```text
hidden -> class logits
hidden -> within-class token logits
```

Possible variants:

1. Exact hierarchical softmax:
   - Train against `log P(class) + log P(token | class)`.
   - Only the target token's class contributes to the within-class term.
   - This reduces compute if implemented carefully.

2. Clustered candidate softmax:
   - Predict top classes.
   - Score only tokens inside selected classes.
   - More approximate and riskier for loss correctness.

3. Shared type/value decoder:
   - Token embedding is composed from a type vector plus a value vector.
   - This targets parameter savings more than softmax compute.

The first version is the cleanest experiment if implemented correctly.

### Phase D: Reinvest savings carefully

If clustered prediction saves parameters or compute, reinvest only after proving the base tax is acceptable.

The control sequence should be:

1. Dense baseline at same vocab.
2. Same-vocab clustered/type model.
3. Budget-reinvested clustered/type model.

The important quantity is:

```text
net_gain = benefit_from_reinvestment - type_factorization_tax
```

This mirrors the qMLP question. A type system that saves parameters but hurts BPB is not a win.

## Why this may matter for the qMLP route

The qMLP experiments showed that parameter savings can be reinvested into vocabulary size, but larger vocab is not free. The `sp16384` simple-stack result stayed under 16 MB but lost to `sp8192`, likely because the larger embedding/head/softmax path slowed training and made optimization harder.

Learned types are interesting because they may offer a different reinvestment route:

- keep or moderately expand vocab
- avoid a full flat-vocab prediction bottleneck
- use token structure to make prediction easier
- preserve the compression benefit of richer tokenization without paying the full softmax tax

That makes learned types complementary to qMLP, not necessarily a replacement.

## Risks

- The implementation may save parameters but not wall-clock time if it still materializes full logits.
- Hierarchical or clustered losses can be numerically or statistically worse than flat softmax.
- Bad clusters may increase BPB even if they look interpretable.
- Token classes learned from one model/vocab may not transfer cleanly to another.
- Exact comparison requires careful artifact-size accounting and seed replication.

## Recommended first concrete test

Start with the 4096 or 8192 vocab, not 1024.

The 1024 vocab is too dominated by bytes and fragments. The 4096 vocab has enough whole words and entity-like pieces to make clustering meaningful, while still being small enough to debug. The 8192 vocab is closer to the current qMLP winner path.

First target:

1. Train or reuse a dense 4096/8192 baseline.
2. Extract token embedding/head rows.
3. Build `K=128` discovered token classes.
4. Inspect class balance and examples.
5. Implement exact hierarchical softmax for the same vocab.
6. Compare against the dense same-vocab baseline on A40.

Only if same-vocab type prediction has a small enough tax should we test budget reinvestment.

## Decision rule

This route is worth pursuing only if it moves us closer to answering:

```text
Can a structured token prediction head beat the best dense/qMLP model under the same 16 MB artifact cap?
```

The type labels are a tool, not the goal. Human-readable classes are helpful for debugging, but BPB under the competition constraints is the deciding metric.
