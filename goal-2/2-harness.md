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

- Phase 3 dense/qMLP package-smoke matrix over ready CaseOps vocabs;
- Phase 4 three-seed dense/qMLP A40 benchmark matrix;
- Phase 5 matrix QA and reruns.

In `HARNESS_MODE=benchmark`, `MAX_WALLCLOCK_SECONDS=600` caps the training loop,
not the full Slurm job. End-to-end job time also includes validation, EMA,
serialization, GPTQ calibration/quantization, brotli packaging, and metrics
parsing.

## Current Assumptions

- The active plan is `goal-2/0-plan.md`.
- The old `goal-1` scripts are reference material only.
- CaseOps `sp1024`, `sp2048`, and `sp4096` exist remotely under `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp<VOCAB>/`.
- CaseOps `sp8192` exists remotely under `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/`.
- CaseOps `sp16384` exists remotely under `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/`.
- All six target CaseOps vocabs are ready for the A40 smoke/benchmark matrix,
  including user-added `sp32768`.
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

Status: complete

Evidence:

- Local syntax checks passed:
  - `bash -n goal-2/2-a40-harness.sbatch`
  - `python3 -m py_compile goal-2/2-parse-metrics.py`
- Remote syntax checks passed with the controlled py311 venv.
- A40 smoke job `20485638` ran on `cn-r-2`, Slurm state `COMPLETED`, exit code `0:0`, elapsed `00:28:55`.
- Smoke configuration:
  - `HARNESS_MODE=smoke`
  - `MODEL_VARIANT=dense`
  - `VOCAB_SIZE=8192`
  - `SEED_VALUE=42`
  - `TTT_ENABLED=0`
  - `DOCUMENT_PACKING=0`
  - `TORCH_COMPILE=0`
  - `FUSED_MLP_ENABLED=0`
  - `FUSED_CE_ENABLED=0`
- The smoke used `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/` and found 80 train shards, one validation shard, and one validation-byte sidecar.
- Key smoke metrics:
  - model params: `35945658`
  - train steps: `2`
  - pre-quant BPB: `4.17567098`
  - quantized no-TTT BPB: `4.17580903`
  - quantized model+brotli: `15880682` bytes
  - total submission size: `15912502` bytes
  - peak memory: `7756 MiB` allocated / `8090 MiB` reserved
- The run wrote `COMPLETE.txt`, `metrics.env`, `metrics.json`, and `metrics.tsv`.

Artifacts:

- Harness: `goal-2/2-a40-harness.sbatch`
- Parser: `goal-2/2-parse-metrics.py`
- Run root: `/nfs/hpc/share/peterj29/pg/runs/goal2-phase2-lean-a40/20485638/`

New facts:

- Dense CaseOps `sp8192` package smoke is under the 16 MB artifact cap.
- The harness can run the A40-friendly record-stack path with TTT disabled and parse post-quant no-TTT BPB.
- The original harness parsed metrics before writing `COMPLETE.txt`, causing `complete_file=0` in the successful smoke metrics. The harness was patched to re-run the parser after writing `COMPLETE.txt` so future successful runs report `complete_file=1`.

Decision:

- Proceed to Phase 3 package-smoke matrix for every ready dense/qMLP CaseOps vocab cell.
