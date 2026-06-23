# Phase 8: Seed Replication

Date drafted: 2026-06-22

## Overview

Replicate the winning Phase 7 `sp4096` matrix qMLP benchmark with additional seeds before spending more compute on `sp8192`, width/depth, or H100/H200 confirmation.

Phase 7 produced a strong single-seed result, but one 10-minute benchmark can still be lucky. This phase checks whether the `sp4096` improvement survives seed variation.

## Candidate To Replicate

Use the exact Phase 7 benchmark shape:

```text
QUAT_MLP=1
QUAT_MLP_IMPL=matrix
VOCAB_SIZE=4096
DATA_PATH=/nfs/hpc/share/peterj29/pg/data-exports/sp4096-80/datasets/fineweb10B_sp4096
TOKENIZER_PATH=/nfs/hpc/share/peterj29/pg/data-exports/sp4096-80/tokenizers/fineweb_4096_bpe.model
ITERATIONS=20000
WARMUP_STEPS=20
TRAIN_LOG_EVERY=20
VAL_LOSS_EVERY=0
TRAIN_BATCH_TOKENS=524288
VAL_BATCH_SIZE=524288
MAX_WALLCLOCK_SECONDS=600
```

Seed 42 reference from Phase 7:

```text
job_id=20480898
steps=352
step_avg=1707.44ms
roundtrip_val_bpb=1.55222627
model_params=11554888
```

Dense baseline reference from Phase 3:

```text
roundtrip_val_bpb=1.58081095
steps=379
step_avg=1587.08ms
model_params=17059912
```

## Implementation Steps

1. Add a reusable seed benchmark Slurm script.

The script should take:

```text
SEED_VALUE=<seed>
```

and otherwise keep the exact Phase 7 benchmark settings.

2. Run seeds one at a time.

Start with:

```text
SEED_VALUE=0
SEED_VALUE=1
```

Do not submit a job array or multiple simultaneous GPU jobs. Use one A40 at a time.

3. Record each seed.

For each seed, collect:

- Slurm state, exit code, elapsed time, node, and allocation;
- `model_params`;
- steps completed;
- `step_avg`;
- final `val_bpb`;
- roundtrip `val_bpb`;
- peak GPU memory;
- int8+zlib artifact size;
- artifact directory.

4. Decide whether the result is robust enough.

Compare each seed against:

- dense `sp1024` Phase 3 roundtrip `val_bpb=1.58081095`;
- qMLP `sp4096` seed 42 roundtrip `val_bpb=1.55222627`.

## Completion Requirements

This phase is complete when:

- at least two additional seed jobs reach terminal state;
- each completed seed has log and artifact facts recorded;
- the docs say whether `sp4096` qMLP is robustly ahead of dense `sp1024`;
- `goal/0-plan.md` and this file are updated;
- the next decision is explicit: try `sp8192`, replicate more, integrate into stronger stack, or stop.

## Failure and Fallback Rules

- If A40 scheduling is blocked, run only one seed when available and document the blocker.
- If a seed fails mechanically, fix the script before running another seed.
- If one seed loses badly to dense, run one more seed before abandoning the path.
- If both extra seeds beat dense, promote `sp4096` qMLP to the next phase.
- Do not change tokenizer, data, width, depth, learning rates, attention, or quantization in this phase.

## Result

Status: complete as of 2026-06-22 07:28 PDT.

Evidence:

- Added reusable seed benchmark script `goal/8-seed.sbatch`.
- Staged remote script at:

```text
/nfs/hpc/share/peterj29/pg/runs/phase8-sp4096-seeds/phase8-seed.sbatch
```

- Seed `0` benchmark job `20480958` ran on `cn-r-3`, partition `share`, one A40, 2 CPUs, 24G RAM, and completed with state `COMPLETED`, exit code `0:0`, elapsed `00:12:44`.
- Seed `1` benchmark job `20480988` ran on `cn-r-6`, partition `share`, one A40, 2 CPUs, 24G RAM, and completed with state `COMPLETED`, exit code `0:0`, elapsed `00:14:25`.
- Seed `0` used the exact Phase 7 benchmark shape:

```text
QUAT_MLP=1
QUAT_MLP_IMPL=matrix
VOCAB_SIZE=4096
TRAIN_BATCH_TOKENS=524288
VAL_BATCH_SIZE=524288
MAX_WALLCLOCK_SECONDS=600
SEED=0
```

- Seed `1` used the same shape with `SEED=1`.

- Seed `0` benchmark facts:

```text
model_params:11554888
steps:353
step_avg:1700.79ms
final_val_bpb:1.5419
roundtrip_val_bpb:1.54759284
peak memory allocated: 13443 MiB reserved: 13454 MiB
Serialized model int8+zlib: 9948374 bytes
Total submission size int8+zlib: 9999583 bytes
```

- Seed `1` benchmark facts:

```text
model_params:11554888
steps:352
step_avg:1708.45ms
final_val_bpb:1.5534
roundtrip_val_bpb:1.55872027
peak memory allocated: 13443 MiB reserved: 13454 MiB
Serialized model int8+zlib: 9942683 bytes
Total submission size int8+zlib: 9993892 bytes
```

- Comparison:

```text
dense_sp1024_phase3_roundtrip_val_bpb=1.58081095
qmlp_sp4096_seed42_roundtrip_val_bpb=1.55222627
qmlp_sp4096_seed0_roundtrip_val_bpb=1.54759284
qmlp_sp4096_seed1_roundtrip_val_bpb=1.55872027
qmlp_sp4096_mean_roundtrip_val_bpb_seed42_0_1=1.55284646
```

- Seed `0` improves over dense `sp1024` by about `0.0332` BPB and improves over the Phase 7 seed `42` run by about `0.0046` BPB.
- Seed `1` improves over dense `sp1024` by about `0.0221` BPB, though it is worse than seed `42` by about `0.0065` BPB.
- The three-run `sp4096` qMLP mean improves over dense `sp1024` by about `0.0280` BPB.

Artifacts:

- `/nfs/hpc/share/peterj29/pg/runs/phase8-sp4096-seeds/20480958/`
- `/nfs/hpc/share/peterj29/pg/runs/phase8-sp4096-seeds/20480988/`

New facts:

- The `sp4096` qMLP win is not isolated to seed `42`; both additional seeds beat dense `sp1024`.
- Throughput is stable across the three runs: seed `42` ran `352` steps at `1707.44ms/step`, seed `0` ran `353` steps at `1700.79ms/step`, and seed `1` ran `352` steps at `1708.45ms/step`.
- Artifact size remains near the challenge budget path: seed `0` total int8+zlib submission size was `9999583` bytes and seed `1` was `9993892` bytes.
- Phase 8 satisfies its completion requirements: at least two additional seed jobs reached terminal state, each has recorded facts, and both support the same decision.

Decision:

- Promote `sp4096` matrix qMLP as the current best simple candidate.
- Next phase should test whether `sp8192` provides more BPB improvement while staying within qMLP's saved-parameter budget and acceptable A40 throughput.
- Do not move to H100/H200 yet; one more A40 vocabulary point is cheaper and directly answers the reinvestment question.
