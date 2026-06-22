# Phase 10: sp8192 Seed Replication

Date drafted: 2026-06-22

## Overview

Replicate the Phase 9 `sp8192` matrix qMLP result with additional seeds on A40 before treating it as the current simple-stack qMLP candidate.

Phase 9 made `sp8192` the strongest simple-stack qMLP result so far, but it is only one seed. This phase checks whether the `sp8192` improvement survives seed variation.

## Why This Matters

The revised plan no longer treats `qMLP sp4096 > dense sp1024` as final evidence. The current question is whether qMLP can help produce the best under-16MB configuration against dense and record-stack controls.

This phase does not answer the record-stack question by itself. It decides which simple-stack qMLP candidate should enter the next simple vocab probe and later controls.

## Current Facts

Dense `sp1024` Phase 3:

```text
roundtrip_val_bpb=1.58081095
steps=379
step_avg=1587.08ms
model_params=17059912
```

Replicated qMLP `sp4096`:

```text
seed42_roundtrip_val_bpb=1.55222627
seed0_roundtrip_val_bpb=1.54759284
seed1_roundtrip_val_bpb=1.55872027
mean_roundtrip_val_bpb=1.55284646
```

qMLP `sp8192` seed 42 from Phase 9:

```text
job_id=20482977
roundtrip_val_bpb=1.52530269
steps=336
step_avg=1789.26ms
model_params=13652040
total_int8_zlib_submission=11602814 bytes
```

## Candidate To Replicate

Use the exact Phase 9 benchmark shape:

```text
QUAT_MLP=1
QUAT_MLP_IMPL=matrix
VOCAB_SIZE=8192
DATA_PATH=/nfs/hpc/share/peterj29/pg/data-exports/sp8192-80/datasets/fineweb10B_sp8192
TOKENIZER_PATH=/nfs/hpc/share/peterj29/pg/data-exports/sp8192-80/tokenizers/fineweb_8192_bpe.model
ITERATIONS=20000
WARMUP_STEPS=20
TRAIN_LOG_EVERY=20
VAL_LOSS_EVERY=0
TRAIN_BATCH_TOKENS=524288
VAL_BATCH_SIZE=524288
MAX_WALLCLOCK_SECONDS=600
```

## Implementation Steps

1. Add a reusable seed benchmark Slurm script.

The script should require:

```text
SEED_VALUE=<seed>
```

and otherwise preserve the Phase 9 settings.

2. Run at least two additional seeds in parallel when schedulable.

Start with:

```text
SEED_VALUE=0
SEED_VALUE=1
```

Submit both seeds concurrently if live scheduler checks and current resource limits allow it. For this phase, two concurrent A40 jobs is acceptable. If Slurm or account limits block parallel execution, run the maximum schedulable subset and document the fallback.

3. Record each seed.

For each seed, collect:

- Slurm state, exit code, elapsed time, node, and allocation;
- GPU name and host;
- `model_params`;
- steps completed;
- `step_avg`;
- final `val_bpb`;
- roundtrip `val_bpb`;
- peak GPU memory;
- int8+zlib artifact size;
- total int8+zlib submission size;
- artifact directory.

4. Decide whether `sp8192` is robust enough.

Compare each seed against:

- dense `sp1024` Phase 3 roundtrip `val_bpb=1.58081095`;
- qMLP `sp4096` replicated mean `val_bpb=1.55284646`;
- qMLP `sp8192` seed 42 roundtrip `val_bpb=1.52530269`.

## Completion Requirements

This phase is complete when:

- at least two additional `sp8192` seed jobs reach terminal state;
- each completed seed has log and artifact facts recorded;
- the docs say whether `sp8192` is seed-robust relative to `sp4096` and dense `sp1024`;
- `goal/0-plan.md` and this file are updated;
- the next decision is explicit: keep `sp8192`, fall back to `sp4096`, run one more seed, or proceed to `sp16384`.

## Failure and Fallback Rules

- If A40 scheduling blocks parallel execution, run the maximum schedulable subset and document the blocker.
- If a seed fails mechanically, fix the script before running another seed.
- If one seed regresses near or above the `sp4096` mean, run one more seed before deciding.
- If both extra seeds beat the `sp4096` mean and dense `sp1024`, keep `sp8192` as the current simple-stack qMLP candidate.
- If multiple seeds regress, keep `sp4096` as the safer simple-stack qMLP candidate.
- Do not change tokenizer, data, width, depth, learning rates, attention, or quantization in this phase.
- Do not treat robust `sp8192` as final winner evidence; it only selects the simple-stack candidate for later controls.

## Result

Status: pending

Evidence:

- Pending implementation.

Artifacts:

- Pending implementation.

New facts:

- Pending implementation.

Decision:

- Pending implementation.
