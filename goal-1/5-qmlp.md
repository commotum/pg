# Phase 5: qMLP A40 Benchmark

Date drafted: 2026-06-22

## Overview

Run the first same-wallclock A40 benchmark for the implemented qMLP path and compare it against the Phase 3 dense A40 baseline.

This phase is not about celebrating the parameter reduction. It answers whether the current quaternion MLP implementation can make useful progress on `val_bpb` within the same 600-second training budget.

## Baseline Reference

Use the Phase 3 dense benchmark as the primary comparison:

```text
job_id=20480569
node=cn-r-3
partition=share
gpu=NVIDIA A40
cpus=2
mem=24G
TRAIN_BATCH_TOKENS=524288
VAL_BATCH_SIZE=524288
WARMUP_STEPS=20
MAX_WALLCLOCK_SECONDS=600
SEED=42
model_params=17059912
steps=379
step_avg=1587.08ms
final_val_bpb=1.5584
roundtrip_val_bpb=1.58081095
int8_zlib_artifact=9265169 bytes
total_int8_zlib_submission=9312855 bytes
```

Use Phase 4 correctness smoke only as an implementation sanity reference:

```text
model_params=9982024
saved_params=7077888
roundtrip_smoke_val_bpb=4.10327187
smoke_step_avg=401.60ms
```

## Implementation Steps

1. Create a qMLP benchmark Slurm script.

Target the same train/eval settings as Phase 3:

```text
RUN_ID=qmlp_a40_10m_seed42_job<job-id>
QUAT_MLP=1
VOCAB_SIZE=1024
SEED=42
ITERATIONS=20000
WARMUP_STEPS=20
TRAIN_LOG_EVERY=20
VAL_LOSS_EVERY=0
TRAIN_BATCH_TOKENS=524288
VAL_BATCH_SIZE=524288
MAX_WALLCLOCK_SECONDS=600
```

2. Prefer the exact Phase 3 resource shape.

```text
partition=share
constraint=a40
gpus=1
cpus-per-task=2
mem=24G
walltime=25m
```

3. Run `srun --test-only` before submitting.

If the exact Phase 3 shape is blocked by `QOSGrpCpuLimit`, test this fallback:

```text
partition=share
constraint=a40
gpus=1
cpus-per-task=1
mem=16G
walltime=25m
```

The fallback is acceptable for forward progress, but the result must be labeled provisional because CPU count differs from the dense reference.

4. Submit only one benchmark job.

Do not keep duplicate pending jobs queued. If the exact-shape job is canceled and replaced by the fallback, record both job IDs and the reason.

5. Monitor the job to completion.

Collect:

- `sacct` state, exit code, elapsed time, node, and allocation;
- `manifest.txt`;
- `gpu.txt`;
- `python.txt`;
- `train.log`;
- `work-files.txt`;
- final model artifacts.

6. Compare against Phase 3.

Record:

- qMLP steps completed in 600 seconds;
- qMLP `step_avg`;
- final and roundtrip `val_bpb`;
- `model_params`;
- peak GPU memory;
- fp32 and int8+zlib model sizes;
- eval time;
- whether the result is exact-shape or provisional fallback.

## Completion Requirements

This phase is complete when:

- one qMLP A40 benchmark has reached a terminal Slurm state;
- the terminal state, exit code, and elapsed time are recorded;
- the run log includes `quat_mlp:True`, `model_params`, completed steps, final `val_bpb`, and roundtrip `val_bpb`;
- artifact sizes are recorded;
- the comparison against Phase 3 is recorded in this file;
- `goal/0-plan.md` is updated with the Phase 5 result;
- the next decision is explicit: optimize qMLP implementation, reinvest saved parameters, rerun for fairness, or stop the qMLP path.

## Failure and Fallback Rules

- If the job fails before training starts, capture the exact error and fix only the failing benchmark mechanics.
- If the job OOMs, reduce only the benchmark batch size enough to get a diagnostic run, and mark it non-comparable.
- If the qMLP benchmark is dramatically slower than dense, do not immediately spend the saved parameters; first decide whether the qMLP implementation should be optimized.
- If a 1-CPU fallback is used, do not overclaim the result. It can identify a strong failure or promising signal, but a close result should be rerun with the Phase 3 2-CPU shape.
- Do not change tokenizer, vocabulary, width, depth, attention, or quantization in this phase.

## Result

Status: complete as of 2026-06-22 02:26 PDT.

Evidence:

- Exact-shape benchmark was possible after a short `QOSGrpCpuLimit` delay, so no 1-CPU fallback was used.
- Slurm fit check for `share/a40`, 1 GPU, 2 CPUs, 24G RAM, and 25 minutes predicted an immediate start on `cn-r-4`.
- Phase 5 benchmark job `20480606` ran on `cn-r-4`, partition `share`, one `NVIDIA A40`, 2 CPUs, 24G RAM, and 25 minute walltime.
- Job `20480606` completed with Slurm state `COMPLETED`, exit code `0:0`, elapsed `00:17:51`.
- The run used the same primary benchmark settings as Phase 3, with `QUAT_MLP=1`:

```text
RUN_ID=qmlp_a40_10m_seed42_job20480606
QUAT_MLP=1
VOCAB_SIZE=1024
SEED=42
ITERATIONS=20000
WARMUP_STEPS=20
TRAIN_LOG_EVERY=20
VAL_LOSS_EVERY=0
TRAIN_BATCH_TOKENS=524288
VAL_BATCH_SIZE=524288
MAX_WALLCLOCK_SECONDS=600
```

- qMLP benchmark log facts:

```text
model_params:9982024
quat_mlp:True
train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:600.000
step:263/20000 val_loss:3.1467 val_bpb:1.8637 train_time:600437ms step_avg:2283.03ms
stopping_early: wallclock_cap train_time:600437ms step:263/20000
peak memory allocated: 13449 MiB reserved: 13566 MiB
Serialized model: 38930363 bytes
Total submission size: 38980568 bytes
Serialized model int8+zlib: 8332875 bytes
Total submission size int8+zlib: 8383080 bytes
final_int8_zlib_roundtrip_exact val_loss:3.16134623 val_bpb:1.87232731
```

- Phase 3 dense comparison:

```text
dense_steps=379
qmlp_steps=263
dense_step_avg=1587.08ms
qmlp_step_avg=2283.03ms
dense_roundtrip_val_bpb=1.58081095
qmlp_roundtrip_val_bpb=1.87232731
dense_int8_zlib_artifact=9265169 bytes
qmlp_int8_zlib_artifact=8332875 bytes
dense_model_params=17059912
qmlp_model_params=9982024
```

- Deltas:

```text
saved_params=7077888
param_reduction=41.5%
steps_completed_delta=-116
steps_completed_ratio=69.4%
step_time_slowdown=1.44x
roundtrip_bpb_delta=+0.29151636
int8_zlib_artifact_delta=-932294 bytes
```

Artifacts:

- Local Phase 5 plan: `goal/5-qmlp.md`.
- Local Phase 5 Slurm script: `goal/5-qmlp.sbatch`.
- Remote Phase 5 Slurm script: `/nfs/hpc/share/peterj29/pg/runs/phase5-qmlp/phase5-qmlp.sbatch`.
- Remote benchmark artifacts: `/nfs/hpc/share/peterj29/pg/runs/phase5-qmlp/20480606/`.
- Remote benchmark files:
  - `train.log`;
  - `manifest.txt`;
  - `gpu.txt`;
  - `python.txt`;
  - `work-files.txt`;
  - `work/final_model.pt`;
  - `work/final_model.int8.ptz`;
  - `work/logs/qmlp_a40_10m_seed42_job20480606.txt`.

New facts:

- The initial qMLP implementation is correct but not competitive in its naive form.
- qMLP saved `7,077,888` parameters, but those savings did not translate into better BPB at fixed wallclock.
- qMLP completed only `263` benchmark steps versus dense baseline `379` steps under the same 600-second training cap.
- qMLP was about `1.44x` slower per training step than dense on the A40 benchmark shape.
- qMLP roundtrip `val_bpb` was worse by about `0.2915`.
- qMLP reduced the int8+zlib model artifact by only about `0.93 MB` despite saving 41.5% of model parameters.
- qMLP peak memory was slightly higher than dense baseline, `13449 MiB` versus `13129 MiB`, so the naive implementation does not currently buy memory headroom.
- qMLP roundtrip eval time was much slower, `103987ms` versus dense baseline `55294ms`.

Decision:

- Do not proceed directly to vocabulary, width, or depth reinvestment.
- The next phase should optimize or replace the qMLP implementation before spending the saved parameter budget.
- The most likely issue is implementation inefficiency: each quaternion projection currently expands into many small `F.linear` calls, which hurts throughput and may also weaken optimization dynamics.
- A reasonable next phase is a fused-real-matrix qMLP experiment that builds the equivalent dense Hamilton block matrix at runtime or stores a packed weight layout that lowers through fewer matmuls. It must preserve the same learned degrees of freedom and parameter count before rerunning the 10-minute A40 benchmark.

Follow-up from Phase 6:

- The matrix implementation was added behind `QUAT_MLP_IMPL=matrix` and passed equivalence checks.
- Matrix qMLP provisional A40 benchmark job `20480636` reached `368` steps and `step_avg:1630.81ms`, so the main Phase 5 speed failure was implementation overhead.
- Matrix qMLP still trailed dense quality with provisional roundtrip `val_bpb:1.64863035` versus dense Phase 3 `1.58081095`; reinvestment remains gated on an exact matrix benchmark and possibly a narrow qMLP quality phase.
