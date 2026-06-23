# Phase 2: Lean A40 Harness

Date drafted: 2026-06-23

## Overview

Create a reusable single-A40 harness for lean record-style Parameter Golf screening.

This phase starts from the proven 04-23 CaseOps record-stack script, but removes the A40-hostile parts from the default loop:

- no full phased TTT;
- no Torch compile by default;
- no document packing by default;
- no fused MLP by default;
- no fused CE by default;
- one A40 GPU;
- post-quant no-TTT `diagnostic quantized` BPB is the primary score.

## Why This Matters

The previous goal showed that full phased TTT on A40 takes roughly an hour and is a bad default benchmark for lean exploration. This harness creates a matched dense/qMLP path that can test model and vocab ideas without accidentally entering the fat eval path.

The harness is the shared base for:

- Phase 3 dense CaseOps `sp8192`;
- Phase 4 qMLP CaseOps `sp8192`;
- Phase 5 qMLP vocab ladder over `1024`, `2048`, `4096`, `8192`, and `16384`.

## Current Assumptions

- The active plan is `goal-2/0-plan.md`.
- The old `goal-1` scripts are reference material only.
- CaseOps `sp8192` exists remotely under `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/`.
- CaseOps `sp16384` exists remotely under `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/`.
- CaseOps `sp1024`, `sp2048`, and `sp4096` still need exports before their qMLP ladder runs.
- The local 04-23 record `train_gpt.py` has SDPA fallback and qMLP support.

## Implementation Steps

1. Add one parameterized Slurm harness.

The harness must support:

- `MODEL_VARIANT=dense`;
- `MODEL_VARIANT=qmlp`;
- `VOCAB_SIZE=...`;
- `HARNESS_MODE=smoke`;
- `HARNESS_MODE=benchmark`;
- explicit `DATA_PATH`, `TOKENIZER_PATH`, and `CASEOPS_ROOT` overrides.

2. Make lean defaults explicit.

Defaults must include:

```text
TTT_ENABLED=0
PHASED_TTT_ENABLED=0
DOCUMENT_PACKING=0
TORCH_COMPILE=0
FUSED_MLP_ENABLED=0
FUSED_CE_ENABLED=0
COMPRESSOR=brotli
```

3. Record benchmark-contract metadata.

Every run should write:

- `manifest.txt`;
- `gpu.txt`;
- `command.txt`;
- `train.log`;
- `work-files.txt`;
- `metrics.tsv`;
- `metrics.env`;
- `COMPLETE.txt` on success.

4. Add a log parser.

The parser should extract at least:

- model params;
- training stop step;
- train time;
- peak memory;
- pre-quant BPB;
- diagnostic quantized BPB;
- quantized model bytes;
- total submission bytes;
- whether TTT accidentally ran.

5. Validate locally.

Run shell syntax checks and Python syntax checks locally. Do not run training locally.

6. Run one A40 package smoke.

Submit one smoke after the scripts are staged remotely:

```text
HARNESS_MODE=smoke
MODEL_VARIANT=dense
VOCAB_SIZE=8192
SEED_VALUE=42
```

This smoke should reach package output and produce `diagnostic quantized` BPB. It is not a serious BPB benchmark.

## Expected Artifacts

- `goal-2/2-a40-harness.sbatch`
- `goal-2/2-parse-metrics.py`
- `goal-2/2-harness.md`
- remote run directory under `/nfs/hpc/share/peterj29/pg/runs/goal2-phase2-lean-a40/`

## Completion Requirements

This phase is complete when:

- the harness script can select dense or qMLP by env flag;
- the harness script can select CaseOps vocab by env flag;
- the harness defaults to no full TTT;
- the harness records the benchmark-contract metadata;
- the parser writes machine-readable metrics;
- local syntax checks pass;
- one A40 smoke reaches package output and writes `metrics.tsv`.

## Failure and Fallback Rules

- If the A40 smoke fails because `sp8192` data is missing, stop and re-run Phase 1 inventory/export rather than changing the harness goal.
- If the A40 smoke OOMs, reduce smoke `TRAIN_BATCH_TOKENS` first; do not switch to full TTT or H100.
- If fused CE fails, keep `FUSED_CE_ENABLED=0` and record the compatibility fact.
- If qMLP fails in the harness, keep the dense path working and record qMLP as the next fix.
- Do not run full phased TTT as part of this phase.

## Result

Status: in progress

Evidence:

- Phase file created.

Decision:

- Implement the Slurm harness and parser next.

