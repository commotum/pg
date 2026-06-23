# Lean A40 Plan for qMLP Parameter Golf Experiments

Date drafted: 2026-06-22

## Objective

Build a single-GPU A40 screening implementation from the best A40-friendly Parameter Golf ideas, then test whether matrix qMLP improves BPB across CaseOps vocabulary sizes.

The primary question is:

```text
Under a lean single-A40 screening setup, does qMLP improve best-under-16MB post-quant no-TTT BPB, and which CaseOps vocab size should be escalated to H100/FA3?
```

This plan intentionally avoids full phased TTT on A40. Full TTT is legal when score-first and within eval budget, but it is a fat evaluation method and was too slow on A40 to be a good exploration benchmark.

## Scope

In scope:

- single-GPU A40 training/eval/package runs;
- CaseOps tokenizer/data preparation;
- vocab sizes `1024`, `2048`, `4096`, `8192`, and `16384`;
- dense baseline at the common optimized CaseOps `sp8192` point;
- qMLP variants at all listed vocab sizes;
- lean record-track ideas that transfer reasonably to A40:
  - CaseOps and original-byte BPB sidecars;
  - SP8192-style vocab defaults;
  - qMLP;
  - XSA;
  - SmearGate;
  - BigramHash or n-gram features if already local and cheap;
  - sparse attention gates / attention output gates;
  - partial RoPE and QK gain;
  - GPTQ/int6-int7 package checks;
  - SDPA/eager compatibility path.

Out of the A40 loop unless explicitly approved:

- full phased TTT benchmarks;
- H100/FA3 performance paths;
- 8xH100 reproduction;
- broad full-wallclock hyperparameter sweeps;
- lrzip/per-group compression if it dominates iteration time.

## Why SP8192 First

The mature record-track entries repeatedly converged on `sp8192`, especially with CaseOps. It is the best default baseline point for an A40-friendly record-style stack.

For this goal:

- dense baseline starts at CaseOps `sp8192`;
- qMLP is tested at `1024`, `2048`, `4096`, `8192`, and `16384`;
- dense lower-vocab controls are secondary and should only run if they clarify whether qMLP is winning for architectural reasons or just vocab/package effects.

## CaseOps Vocabulary Policy

CaseOps vocab is not the same as the simple-stack SentencePiece vocab.

For each `VOCAB_SIZE`, prepare a CaseOps tokenizer by:

1. reading the selected FineWeb docs;
2. applying `encode_lossless_caps_v2`;
3. training SentencePiece BPE with CaseOps reserved control symbols;
4. using byte fallback, split digits, no dummy prefix, and reserved IDs;
5. exporting train shards, validation shards, and validation-byte sidecars with `prepare_caseops_data.py`;
6. scoring BPB on original bytes using the byte sidecars.

Dense and qMLP runs with the same `VOCAB_SIZE` must use the exact same CaseOps tokenizer model, vocab file, train shards, validation shards, and byte sidecars. The only intended difference between a paired dense/qMLP run is the model architecture flag.

Different vocab sizes require different CaseOps tokenizer/data exports:

```text
caseops-sp1024
caseops-sp2048
caseops-sp4096
caseops-sp8192
caseops-sp16384
```

## Benchmark Contract

Default A40 benchmark metrics:

- pre-quant `val_bpb`;
- post-quant no-TTT `val_bpb`;
- total submission bytes;
- quantized model bytes;
- training steps;
- ms/step;
- peak GPU memory;
- host/GPU;
- seed;
- command/config;
- run path and job ID.

Do not use final phased TTT BPB as the primary A40 score. If TTT is touched at all, use a tiny legality smoke only.

## Operating Rules

- Local Mac is for editing and planning.
- OSU flip/HPC submit is for Slurm control only.
- Slurm compute nodes handle CPU exports, GPU smokes, and benchmarks.
- No training, tokenizer export, GPU diagnostics, or material compute on submit nodes.
- Use CPU Slurm allocations with at least 16 CPUs for full CaseOps exports unless a smaller smoke is intentional.
- Parallelize independent seeds and vocab package smokes when scheduler/account limits allow.
- Preserve dependency order: data -> smoke -> benchmark.
- Ask before H100/H200, multi-GPU, long jobs, destructive changes, or broad sweeps.

## Phase Plan

### Phase 0: Inventory

Goal: capture the current repo, remote state, existing CaseOps exports, and usable Slurm partitions without starting new training.

Actions:

1. Check local and remote git status.
2. List existing CaseOps exports for `1024`, `2048`, `4096`, `8192`, and `16384`.
3. List existing qMLP patches and record-stack patches.
4. Check user queue and live A40 availability.
5. Record whether old `goal-1` jobs are still running, but do not cancel them without approval.

Exit criteria:

- inventory phase file exists;
- no compute work has run on submit nodes;
- next data/export phase knows what can be reused.

### Phase 1: CaseOps Data Matrix

Goal: prepare or verify CaseOps tokenizer/data exports for the vocab ladder.

Actions:

1. Reuse completed CaseOps exports when valid.
2. For missing vocab sizes, run CPU Slurm exports with 16+ CPUs.
3. Use bounded smoke exports before full 80-shard exports when a tokenizer path or script path is uncertain.
4. Verify tokenizer model, vocab file, manifest, train shards, validation shards, and byte sidecars.
5. Confirm that paired dense/qMLP runs at each vocab size will point to the same data paths.

Vocab targets:

```text
1024
2048
4096
8192
16384
```

Exit criteria:

- every target vocab is either ready or has a captured blocker;
- `sp8192` is ready before dense baseline benchmarking starts;
- data paths are recorded in the phase file.

### Phase 2: Lean A40 Harness

Goal: create a reusable single-A40 harness for the lean record-style stack.

Default lean settings:

```text
TTT_ENABLED=0
PHASED_TTT_ENABLED=0
DOCUMENT_PACKING=0 unless proven safe
TORCH_COMPILE=0 unless proven safe
FUSED_MLP_ENABLED=0 unless proven safe
FUSED_CE_ENABLED=0 unless proven safe
attention=SDPA/eager fallback
single GPU
post-quant no-TTT BPB is primary
```

Record-style lean features to include if already local and cheap:

- CaseOps;
- XSA;
- SmearGate;
- sparse attention gate / attention output gate;
- partial RoPE;
- QK gain;
- quantization/package path.

Exit criteria:

- one script or small set of scripts can run dense or qMLP by env flags;
- script records all benchmark contract metrics;
- smoke run reaches package output on A40.

### Phase 3: Dense SP8192 Baseline

Goal: establish the lean A40 dense baseline at the common CaseOps `sp8192` point.

Actions:

1. Run package/path smoke.
2. Run seed `42` A40 benchmark.
3. If stable, run seeds `0` and `1` in parallel where scheduler limits allow.
4. Record pre-quant, post-quant no-TTT BPB, package size, memory, steps, and step time.

Exit criteria:

- dense CaseOps `sp8192` has at least one complete run;
- preferably three seeds exist before interpreting small deltas;
- baseline is under 16 MB or the oversize condition is documented.

### Phase 4: qMLP Harness Check at SP8192

Goal: add qMLP to the same lean A40 stack without changing vocab or unrelated settings.

Actions:

1. Enable `QUAT_MLP=1` and `QUAT_MLP_IMPL=matrix`.
2. Use the exact same CaseOps `sp8192` tokenizer/data as Phase 3.
3. Run package/path smoke.
4. Run matched seed batch.
5. Compare qMLP `sp8192` against dense `sp8192`.

Exit criteria:

- qMLP path initializes and packages;
- same-vocab qMLP delta is recorded;
- package-size savings are measured.

### Phase 5: qMLP Vocab Ladder

Goal: test qMLP at CaseOps vocab sizes `1024`, `2048`, `4096`, `8192`, and `16384`.

Actions:

1. Run package/path smokes for all target qMLP vocab sizes, parallelized where safe.
2. Stop any vocab size that exceeds 16 MB or has path/data errors.
3. Benchmark serious candidates on seed `42`.
4. Run seeds `0` and `1` for candidates that beat or nearly match the current best.
5. Do not benchmark larger vocab one point at a time beyond `16384` unless the user redirects.

Exit criteria:

- all target qMLP vocab sizes have package-smoke outcomes;
- at least `8192` and any promising neighbor have benchmark outcomes;
- best qMLP vocab is selected by post-quant no-TTT BPB and package size.

### Phase 6: Dense Budget Controls

Goal: determine whether qMLP wins because of architecture or because dense was underfilled.

Actions:

1. Use cheap package smokes to estimate dense vocab sizes that fit under 16 MB.
2. Do not run full dense benchmarks at every vocab size by default.
3. Benchmark dense near-budget controls only if they materially clarify the qMLP result.

Exit criteria:

- either dense budget controls are not needed, with rationale;
- or one or more dense controls are benchmarked and compared.

### Phase 7: A40 Decision

Goal: decide what, if anything, deserves H100/FA3 confirmation.

Actions:

1. Summarize dense `sp8192`, qMLP `sp8192`, and the best qMLP vocab.
2. Separate lean gains from fat dependencies.
3. Identify whether the candidate should be ported to H100/FA3 with full legal eval.
4. Update all `goal-2` docs.

Exit criteria:

- keep/modify/abandon qMLP decision exists;
- H100 candidate command is drafted only if A40 evidence justifies it;
- no H100/H200 work is launched without explicit approval.

## Current Starting Assumptions

- `goal-1` has been renamed and is no longer the active loop.
- qMLP code and CaseOps export scripts exist from prior work, but must be re-inventoried.
- CaseOps `sp8192` and `sp16384` exports may already exist remotely.
- `sp8192` is the most common optimized record-track vocab and should be the dense default.
- Full A40 phased TTT is too slow to be a useful default benchmark.

## Success Criteria

The goal succeeds if it produces:

- a reusable lean A40 benchmark harness;
- a dense CaseOps `sp8192` baseline;
- matched qMLP results across the requested vocab ladder;
- a defensible best qMLP candidate or a defensible decision to abandon qMLP;
- clear docs separating A40 screening evidence from record-compliant evidence.

