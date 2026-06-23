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

Status: complete as of 2026-06-22 11:22 PDT.

Evidence:

- Added reusable seed benchmark script `goal/10-seed.sbatch`.
- Staged remote script at:

```text
/nfs/hpc/share/peterj29/pg/runs/phase10-sp8192-seeds/phase10-seed.sbatch
```

- Seed `0` benchmark job `20483042` and seed `1` benchmark job `20483043` were submitted concurrently.
- Both jobs ran in parallel on `cn-r-3`, partition `share`, with one A40, 2 CPUs, and 24G RAM per job.
- Seed `0` Slurm state was `COMPLETED`, exit code `0:0`, elapsed `00:14:18`.
- Seed `1` Slurm state was `COMPLETED`, exit code `0:0`, elapsed `00:14:14`.
- Seed `0` used A40 UUID `GPU-3ea3bfa5-42e5-767d-1df8-53592b677d3b`.
- Seed `1` used A40 UUID `GPU-c6602d49-5711-7014-67cb-9216db753042`.
- Both seeds used the exact Phase 9 benchmark shape:

```text
QUAT_MLP=1
QUAT_MLP_IMPL=matrix
VOCAB_SIZE=8192
TRAIN_BATCH_TOKENS=524288
VAL_BATCH_SIZE=524288
MAX_WALLCLOCK_SECONDS=600
```

- Seed `0` benchmark facts:

```text
job_id=20483042
model_params=13652040
steps=337
step_avg=1781.81ms
final_val_bpb=1.5198
roundtrip_val_bpb=1.52474158
peak memory allocated: 13971 MiB reserved: 14026 MiB
Serialized model int8+zlib: 11545409 bytes
Total submission size int8+zlib: 11596618 bytes
```

- Seed `1` benchmark facts:

```text
job_id=20483043
model_params=13652040
steps=337
step_avg=1782.84ms
final_val_bpb=1.5201
roundtrip_val_bpb=1.52563039
peak memory allocated: 13971 MiB reserved: 14026 MiB
Serialized model int8+zlib: 11543981 bytes
Total submission size int8+zlib: 11595190 bytes
```

- Comparison:

```text
dense_sp1024_phase3_roundtrip_val_bpb=1.58081095
qmlp_sp4096_mean_roundtrip_val_bpb_seed42_0_1=1.55284646
qmlp_sp8192_seed42_roundtrip_val_bpb=1.52530269
qmlp_sp8192_seed0_roundtrip_val_bpb=1.52474158
qmlp_sp8192_seed1_roundtrip_val_bpb=1.52563039
qmlp_sp8192_mean_roundtrip_val_bpb_seed42_0_1=1.52522489
```

- Deltas:

```text
sp8192_mean_vs_dense_sp1024=-0.05558606 BPB
sp8192_mean_vs_sp4096_mean=-0.02762157 BPB
```

Artifacts:

- `/nfs/hpc/share/peterj29/pg/runs/phase10-sp8192-seeds/20483042/`
- `/nfs/hpc/share/peterj29/pg/runs/phase10-sp8192-seeds/20483043/`

New facts:

- `sp8192` is seed-robust on the simple stack: seed `42`, seed `0`, and seed `1` all beat the replicated `sp4096` mean and dense `sp1024`.
- Parallel seed execution worked under current scheduler/account limits for two concurrent A40 jobs.
- `sp8192` throughput is stable across seeds: `336-337` steps with `1781.81-1789.26ms/step`.
- Artifact size remains safely under 16 MB: both added seeds were about `11.60 MB` total int8+zlib submission size.

Decision:

- Keep `sp8192` as the current simple-stack qMLP candidate.
- Proceed to Phase 11: an `sp16384` initial qMLP probe with export, smoke/package-size gate, and a parallel seed batch only if the smoke gate passes.
- Do not treat `sp8192` as final winner evidence; it still needs dense budget controls and record-stack comparison.
