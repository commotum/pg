---
title: "Record candidate: PR #2130 base + GPTQ_CALIBRATION_BATCHES=32 — val_bpb 1.05651 (3-seed mean) by codemath3000 · Pull Request #2135 · openai/parameter-golf"
source: "https://github.com/openai/parameter-golf/pull/2135"
author:
  - "[[codemath3000]]"
published: 2026-05-01
created: 2026-06-22
description: "Record candidate: PR #1797 base + Token-only n-gram tilt + AsymLogit Rescale + #2060 levers + NUM_PHASES=1 + GPTQ_CALIBRATION_BATCHES=32 val"
tags:
  - "clippings"
---
**val\_bpb: 1.05651** (3-seed mean, std 0.00036) | **val\_loss: 2.31203 nats** (std 0.00078) | **15.95 MB max** | 8xH100 SXM | 600s train / 600s eval

**Improvement over merged PR [#1855](https://github.com/openai/parameter-golf/pull/1855) leaderboard record (1.06107587 BPB):**  
**\-0.00457 BPB / -0.01000 nats**

The -0.01000-nat improvement clears the README's record-acceptance threshold of **0.005 nats** by approximately 2x.

This submission keeps the PR [#2130](https://github.com/openai/parameter-golf/pull/2130) architecture/training stack identical and only changes one knob: **GPTQ\_CALIBRATION\_BATCHES=32** (PR [#2130](https://github.com/openai/parameter-golf/pull/2130) used 16). Every other hyperparameter, env var, and code path is byte-for-byte the PR [#2130](https://github.com/openai/parameter-golf/pull/2130) reproduction command.

## Note on submission timing

This PR was opened at 2026-05-01 (before the deadline) with the full code, environment, and submission scaffold in place. The per-seed run logs and result numbers were filled in afterward. The official README makes the policy explicit: "we're accepting record submissions chronologically depending on their PR creation time."

The PR [#1851](https://github.com/openai/parameter-golf/pull/1851) / PR [#1868](https://github.com/openai/parameter-golf/pull/1868) pair is a direct precedent. PR [#1851](https://github.com/openai/parameter-golf/pull/1851) was opened with incomplete seed results (the headline value sat between the then-current SOTA and the next eventual record, PR [#1855](https://github.com/openai/parameter-golf/pull/1855)). PR [#1868](https://github.com/openai/parameter-golf/pull/1868) later filled in the missing seed data for the same submission. When the package was evaluated, it was treated as record-relevant on PR [#1851](https://github.com/openai/parameter-golf/pull/1851)'s chronological standing, not PR [#1868](https://github.com/openai/parameter-golf/pull/1868)'s: the original PR's creation time was the dispositive timestamp, even though the data needed to substantiate the claim arrived later. The same principle applies here: **PR creation time, not results-posting time, is what establishes chronological standing.** Under this rule, the present submission counts as filed before the deadline and is therefore valid.

## Results

| Seed | Steps | ms/step | Train ms | Pre-quant BPB | Quant BPB | **Post-TTT BPB** | TTT eval s | Artifact bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4,997 | 120.0 | 599,608 | 1.06105556 | 1.06939370 | **1.05679341** | 567.1 | 15,942,822 |
| 42 | 5,001 | 119.9 | 599,672 | 1.06026908 | 1.06867913 | **1.05610947** | 540.1 | 15,947,490 |
| 314 | 4,983 | 120.3 | 599,593 | 1.06091124 | 1.06921334 | **1.05662016** | 567.1 | 15,945,305 |
| **Mean** | **4,994** | **120.1** | **599,624** | **1.06074529** | **1.06909539** | **1.05650768** | **558.1** | **15,945,206** |

3-seed sample std: **0.00035573 BPB / 0.00077803 nats**.

All seeds under the 16,000,000-byte artifact cap and 600s train/eval budgets. Maximum artifact is **15,947,490 bytes** and the maximum eval pass is **567.1s**.

## Statistical significance vs PR [#2130](https://github.com/openai/parameter-golf/pull/2130) (immediate prior candidate)

Both PRs were evaluated on the same seed set {0, 42, 314}. With identical seeds on both sides, the **paired t-test is the appropriate comparison** since each seed pair shares its random initialization, removing the seed-to-seed component of the variance and giving the test the most power. For completeness, we also report **Welch's t-test (unbiased σ)** treating the two sets as independent samples; both tests clear the threshold.

| Test | t-stat | df | p (one-tailed) | Verdict at p<0.25 |
| --- | --- | --- | --- | --- |
| **Paired t-test (primary, same seeds)** | **\-1.48** | **2** | **0.138** | **PASSES** |
| Welch's t-test (unbiased σ, for completeness) | \-0.82 | 2.96 | 0.238 | PASSES |

Per-seed post-TTT BPB (full precision, from each PR's run logs):

| Seed | PR [#2130](https://github.com/openai/parameter-golf/pull/2130) | This submission | Δ |
| --- | --- | --- | --- |
| 0 | 1.05689587 | 1.05679341 | \-0.00010246 |
| 42 | 1.05654656 | 1.05610947 | \-0.00043709 |
| 314 | 1.05664400 | 1.05662016 | \-0.00002384 |

We tie or beat PR [#2130](https://github.com/openai/parameter-golf/pull/2130) on every seed.

## What is new in this submission

A single hyperparameter change on the PR [#2130](https://github.com/openai/parameter-golf/pull/2130) stack: **GPTQ\_CALIBRATION\_BATCHES**, increased from 16 to 32. This doubles the number of GPTQ Hessian calibration batches at the (still in-budget) end of training, giving GPTQ a denser activation estimate and reducing post-quantization error.

| Lever | PR [#2130](https://github.com/openai/parameter-golf/pull/2130) | This submission | Mechanism |
| --- | --- | --- | --- |
| `GPTQ_CALIBRATION_BATCHES` | 16 | 32 | More calibration batches for GPTQ Hessian estimation; better activation statistics, lower post-quant error. |

Everything else (model, optimizer, schedule, TTT, compression, n-gram tilt configuration) is byte-for-byte identical to the PR [#2130](https://github.com/openai/parameter-golf/pull/2130) stack.

## Compliance

This submission inherits compliance directly from PR [#2130](https://github.com/openai/parameter-golf/pull/2130) (which itself stacks on the merged PR [#1797](https://github.com/openai/parameter-golf/pull/1797) / PR [#1855](https://github.com/openai/parameter-golf/pull/1855) lineage). The single lever change does not affect any compliance-relevant code path:

- **C1 causality.** The token-only n-gram tilt is unchanged. Within-word and word-start channels remain disabled (`WITHIN_TAU=99.0, WITHIN_BOOST=0.0, WORD_TAU=99.0, WORD_BOOST=0.0, AGREE_ADD_BOOST=0.0`). Logs confirm `within_gate=0 word_gate=0 agree2plus=0` in all 3 seeds; only the strictly-causal `token_hint` channel fires (`token_gate=628130`). Per merged PR [Record: SP8192 + Muon 0.97 + Legal Score-First TTT — val\_bpb 1.07983 (3-seed mean) #1514](https://github.com/openai/parameter-golf/pull/1514) ruling.
- **C2 normalization.** The closed-form n-gram tilt applies `logit += boost` and renormalizes via softmax. Probability mass is preserved by construction. Unchanged from PR [Record candidate: 1.05670 BPB — token-only n-gram tilt + AsymLogit + #2060 levers + NUM\_PHASES=1 #2130](https://github.com/openai/parameter-golf/pull/2130).
- **C3 score-first TTT.** The `eval_val_ttt_phased` path scores each chunk before applying that chunk's LoRA update. Unchanged from PR [Record candidate: 1.05670 BPB — token-only n-gram tilt + AsymLogit + #2060 levers + NUM\_PHASES=1 #2130](https://github.com/openai/parameter-golf/pull/2130).
- **C4 single L→R pass.** Each validation token contributes exactly one BPB term. Unchanged from PR [Record candidate: 1.05670 BPB — token-only n-gram tilt + AsymLogit + #2060 levers + NUM\_PHASES=1 #2130](https://github.com/openai/parameter-golf/pull/2130).

Eval-wallclock disclosure: total eval times include the n-gram precompute, computed INSIDE the eval timer (`NGRAM_HINT_PRECOMPUTE_OUTSIDE=0`). Logs explicitly show `ngram_hint_precompute_outside: False` and a `precompute_done` line. All 3 seeds well under the 600s eval budget (max 567.1s, mean 558.1s).

GPTQ calibration uses training shards only and runs in the training phase under `GPTQ_RESERVE_SECONDS=0.5` per Issue [#1017](https://github.com/openai/parameter-golf/issues/1017). The doubled calibration count adds a few seconds to the calibration step that the existing `RESERVE_SECONDS` deduction already covers; no validation data is ever accessed during training.

No SLOT, persistent n-gram cache, PPM mixture, logit bias table, or validation-derived precomputation. CaseOps byte sidecar accounting is preserved.

## Architecture and training stack

Identical to PR [#2130](https://github.com/openai/parameter-golf/pull/2130). Summary table for reviewer convenience:

| Component | Setting |
| --- | --- |
| Model | 11 layers, 512d, 8 query heads, 4 KV heads, MLP 4x |
| Tokenizer/data | SP8192 CaseOps lossless caps, byte sidecar accounting (PR [#1729](https://github.com/openai/parameter-golf/pull/1729) / [#1736](https://github.com/openai/parameter-golf/pull/1736) lineage) |
| Recurrence | Layers 3-5 looped at frac=0.35, parallel decoder layer 8+ |
| Gates | BOS-fixed SmearGate (GATE\_WINDOW=12), SparseAttnGate (scale=0.5) |
| Optimizer | Muon on matrix params (LR=0.028), Adam on embedding/scalars (BETA2=0.99) |
| EMA | EMA\_DECAY=0.9965 |
| Quantization | GPTQ int6 matrices, int7 embeddings, LQER asymmetric rank-4 (GROUP=32, TOP\_K=3), GPTQ\_RESERVE\_SECONDS=0.5 |
| Compression | per-group (lrzip + brotli) |
| Eval context | EVAL\_SEQ\_LEN=2560, TTT\_EVAL\_SEQ\_LEN=2560 |
| TTT | Quantized phased LoRA (RANK=80, LR=8e-5, BETA2=0.99, WEIGHT\_DECAY=2.0), score-first, K-off, 1 phase, 2500-doc prefix |
| Logit softcap | AsymLogit Rescale (softcap\_pos, softcap\_neg, init 30.0) |
| Tilt | Token-only n-gram tilt (TOKEN\_ORDER=16, TOKEN\_THRESHOLD=0.800, TOKEN\_BOOST=2.625) |

## Lineage and credits

- PR [Record candidate: 1.05670 BPB — token-only n-gram tilt + AsymLogit + #2060 levers + NUM\_PHASES=1 #2130](https://github.com/openai/parameter-golf/pull/2130) by [@codemath3000](https://github.com/codemath3000) — direct comparison baseline; token-only n-gram tilt + AsymLogit Rescale + [Record: LongCtx No-QV QK5.25 + AsymLogit + LQER g32/top4 + TTT-local 0.80 — 1.05792 BPB 3-seed mean #2060](https://github.com/openai/parameter-golf/pull/2060) levers + NUM\_PHASES=1 stacked on PR [Record: PR #1787 base + Smear Gate + LQER Asym — val\_bpb 1.06157 #1797](https://github.com/openai/parameter-golf/pull/1797) / PR [Record: SP8192 + LQER + Sparse Attn Gate + BOS-Fixed SmearGate + 9-Hparam Greedy Stack — val\_bpb 1.06108 (3-seed mean) #1855](https://github.com/openai/parameter-golf/pull/1855) lineage.
- PR [Record: SP8192 + LQER + Sparse Attn Gate + BOS-Fixed SmearGate + 9-Hparam Greedy Stack — val\_bpb 1.06108 (3-seed mean) #1855](https://github.com/openai/parameter-golf/pull/1855) by [@codemath3000](https://github.com/codemath3000) — current merged leaderboard record.
- PR [Record: PR #1787 base + Smear Gate + LQER Asym — val\_bpb 1.06157 #1797](https://github.com/openai/parameter-golf/pull/1797) by [@dexhunter](https://github.com/dexhunter) — SmearGate and LQER asymmetric rank-4 lineage; direct base of PR [Record candidate: 1.05670 BPB — token-only n-gram tilt + AsymLogit + #2060 levers + NUM\_PHASES=1 #2130](https://github.com/openai/parameter-golf/pull/2130).
- PR [Record: SP8192 + Muon 0.97 + Legal Score-First TTT — val\_bpb 1.07983 (3-seed mean) #1514](https://github.com/openai/parameter-golf/pull/1514) (merged) — token-only n-gram tilt legality precedent.
- PR [Record: SP8192 #1855 Base + Asymmetric Logit Rescale + AWQ-lite — val\_bpb 1.05971 (3-seed mean, full val) #1923](https://github.com/openai/parameter-golf/pull/1923) — AsymLogit Rescale technique.
- PR [Record: LongCtx No-QV QK5.25 + AsymLogit + LQER g32/top4 + TTT-local 0.80 — 1.05792 BPB 3-seed mean #2060](https://github.com/openai/parameter-golf/pull/2060) by [@S0urC10ud](https://github.com/S0urC10ud) — hyperparameter sweep values.
- PR [Record: PR1855/PR1953 base + Progressive context growth (val\_bpb: 1.05759, 3-seed) #2014](https://github.com/openai/parameter-golf/pull/2014) by [@simonbissonnette](https://github.com/simonbissonnette) — PHASED\_TTT\_NUM\_PHASES=1 precedent.
- PR [Record: CaseOps Tokenizer + Tapered WD - val\_bpb 1.0678 (3-seed mean) #1729](https://github.com/openai/parameter-golf/pull/1729) by [@romeerp](https://github.com/romeerp) / [@dexhunter](https://github.com/dexhunter) — CaseOps tokenizer.

The new contribution here is the targeted hyperparameter change (GPTQ\_CALIBRATION\_BATCHES=32), isolated as a clean ablation so reviewers can compare the lever effect against PR [#2130](https://github.com/openai/parameter-golf/pull/2130) directly.

---

## Comments

> **cocohearts** · 2026-05-02
> 
> Update after applying the grace policy: this is now included in leaderboard PR [#2146](https://github.com/openai/parameter-golf/pull/2146).
> 
> The PR/code/scaffold were opened before cutoff, and the post-cutoff commits supplied the run logs/results. I also verified the submitted logs use clean canonical CaseOps data (`train_shards: 80`, no doubled local `datasets/datasets` path, `val_tokens: 47851520`) and token-only n-gram gates (`within_gate=0 word_gate=0 agree2plus=0`). The leaderboard row uses the submitted 3-seed mean 1.05650768.

> **codemath3000** · 2026-05-02
> 
> [@cocohearts](https://github.com/cocohearts) Thanks so much for taking the time to review this. The "scored state must be in by cutoff" criterion isn't in the README. The official README sets the rule: "Since all submissions are public, we're accepting record submissions chronologically depending on their PR creation time." That pins ordering on PR creation, not on when content reached completion.
> 
> PR [#2135](https://github.com/openai/parameter-golf/pull/2135) was opened at 2026-05-01 23:48:57Z (4:48 PM PT, ~11 min before the 5 PM PT cutoff), so it satisfies that test.
> 
> The pre-cutoff commit ([be7420d](https://github.com/openai/parameter-golf/commit/be7420d313586bef947ef2043d0d3065a9dec2f2)) shipped the full code surface: train\_gpt.py, lossless\_caps.py, online\_ngram\_state.c, online\_ngram\_tilt.py, prepare\_caseops\_data.py, the tokenizer .model, requirements.txt, and scaffold README/submission.json. The two post-cutoff commits ([f086a9f](https://github.com/openai/parameter-golf/commit/f086a9fe1c29e15197f35ffb782f387ff8571f8f), [ff90522](https://github.com/openai/parameter-golf/commit/ff905226136eb2506fe9505feff87f58eb6dc4ff)) only added the train\_seed\*.log run outputs and filled result numerics into README.md and submission.json. No methodology, architecture, training script, or tokenizer changed post-cutoff.
> 
> The seed log files added afterward are run outputs of the pre-cutoff code, not new submission content. Anyone with 8xH100 SXM access could clone [be7420d](https://github.com/openai/parameter-golf/commit/be7420d313586bef947ef2043d0d3065a9dec2f2) and regenerate the runs within run-to-run noise. What was "missing" at cutoff wasn't the submission; it was the empirical measurement of a submission that was already public and reproducible.
> 
> Merged precedent for post-creation supplementation: PR [#1851](https://github.com/openai/parameter-golf/pull/1851) was supplemented via PR [#1868](https://github.com/openai/parameter-golf/pull/1868), after PR [#1855](https://github.com/openai/parameter-golf/pull/1855) had already taken over the leaderboard. The combined submission was still accepted as part of the leaderboard record. Applied consistently, the "content complete by cutoff" rule would retroactively reject that combined submission (already part of the merged leaderboard): PR [#1851](https://github.com/openai/parameter-golf/pull/1851)'s scored 3-seed state was completed by PR [#1868](https://github.com/openai/parameter-golf/pull/1868) well after [#1851](https://github.com/openai/parameter-golf/pull/1851)'s own creation and after PR [#1855](https://github.com/openai/parameter-golf/pull/1855) had displaced it.
> 
> Thanks so much again for taking the time to review.