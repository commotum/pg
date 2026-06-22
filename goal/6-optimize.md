# Phase 6: qMLP Implementation Optimization

Date drafted: 2026-06-22

## Overview

Phase 5 showed that the first qMLP implementation is mechanically correct but not competitive: it is about `1.44x` slower per step and about `0.2915` BPB worse after roundtrip than the dense Phase 3 baseline.

This phase tests whether that failure is mainly caused by the naive split implementation's many small matmuls. The optimization target is an equivalent Hamilton-matrix forward path that preserves the same learned parameters and model parameter count, but lowers each quaternion projection through one dense `F.linear`.

## Baseline Facts

Dense Phase 3 reference:

```text
steps=379
step_avg=1587.08ms
roundtrip_val_bpb=1.58081095
model_params=17059912
int8_zlib_artifact=9265169 bytes
```

Naive qMLP Phase 5 reference:

```text
steps=263
step_avg=2283.03ms
roundtrip_val_bpb=1.87232731
model_params=9982024
int8_zlib_artifact=8332875 bytes
```

## Implementation Steps

1. Add a qMLP implementation selector.

Use an environment variable:

```text
QUAT_MLP_IMPL=split
QUAT_MLP_IMPL=matrix
```

`split` should preserve the Phase 5 implementation. `matrix` should construct the equivalent Hamilton block matrix and run a single `F.linear`.

2. Preserve the learned parameterization.

Both implementations must use the same learned tensors:

```text
wr
wi
wj
wk
```

The optimized path must not add learned dense weights or biases.

3. Add correctness checks.

The correctness script should verify:

- `split` and `matrix` produce numerically equivalent outputs for the same weights;
- both variants match the explicit Hamilton matrix;
- parameter count remains `9,982,024`;
- qMLP component parameters remain visible to the existing Muon matrix grouping;
- a toy qMLP model can run forward/backward with `QUAT_MLP_IMPL=matrix`.

4. Run a qMLP matrix smoke.

Use the Phase 4 smoke settings plus:

```text
QUAT_MLP_IMPL=matrix
```

5. Run a qMLP matrix A40 benchmark.

Use the Phase 5 benchmark settings plus:

```text
QUAT_MLP_IMPL=matrix
```

Prefer the exact Phase 3/5 resource shape:

```text
partition=share
constraint=a40
gpus=1
cpus-per-task=2
mem=24G
walltime=25m
```

If this exact shape is blocked by `QOSGrpCpuLimit`, wait briefly. Use a 1-CPU fallback only for smoke, not for the primary benchmark unless the result will be clearly labeled provisional.

## Completion Requirements

This phase is complete when:

- the matrix implementation exists behind `QUAT_MLP_IMPL=matrix`;
- `QUAT_MLP_IMPL=split` remains available for comparison;
- correctness checks pass;
- a matrix qMLP smoke reaches `COMPLETED`;
- a matrix qMLP A40 benchmark reaches a terminal state;
- `goal/0-plan.md` and this file record the smoke and benchmark facts;
- the decision says whether to reopen reinvestment, keep optimizing, or stop the qMLP path.

## Failure and Fallback Rules

- If matrix output is not equivalent to split output, fix signs/layout before any Slurm run.
- If matrix path compiles but is still much slower than dense, do not reinvest saved parameters.
- If matrix path is faster but still much worse BPB, treat parameter sharing as the likely problem and stop or try a very small learning-rate/init adjustment phase.
- If matrix path becomes close in speed and BPB, proceed to the reinvestment grid.
- Do not change tokenizer, vocabulary, width, depth, attention, batch settings, or quantization in this phase.

## Result

Status: complete for the Phase 6 decision as of 2026-06-22 03:19 PDT. Matrix qMLP correctness and smoke are complete, and a provisional 1-CPU A40 benchmark is complete. The exact 2-CPU/24G rerun is still desirable for bookkeeping, but no longer blocks the next phase because the provisional run answers the implementation-speed question.

Evidence:

- Added `QUAT_MLP_IMPL` with two qMLP implementations:
  - `split`: the Phase 5 path using repeated component `F.linear` calls;
  - `matrix`: an equivalent Hamilton block matrix path using one `F.linear` per quaternion projection.
- `QUAT_MLP_IMPL=matrix` preserves the same learned tensors: `wr`, `wi`, `wj`, and `wk`.
- `QUAT_MLP_IMPL=split` remains available for direct comparison.
- Local syntax checks passed:

```text
python3 -m py_compile parameter-golf/train_gpt.py goal/4-qmlp-check.py
bash -n goal/6-smoke.sbatch goal/6-benchmark.sbatch
```

- Remote CPU correctness check passed in the HPC torch venv:

```text
shape_checks=ok
hamilton_equivalence_max_error=1.19e-07
matrix_split_equivalence_max_error=1.19e-07
dense_params=17059912
qmlp_params=9982024
saved_params=7077888
qmlp_muon_matrix_params=72
toy_forward_backward=ok loss=3.463055
```

- A40 smoke scheduling facts:
  - `share/a40`, 1 GPU, 1 CPU, 16G, 20 minutes test-only predicted start on `cn-r-4`;
  - submitted smoke job `20480617`;
  - job `20480617` stayed pending on `QOSGrpCpuLimit` with no start estimate;
  - `preempt/a40` test-only estimated `2026-06-22T04:28:37`;
  - `ampere/a40` test-only estimated `2026-06-22T06:56:41`;
  - job `20480617` was canceled while still pending;
  - final `squeue -u peterj29` showed no remaining queued or running jobs.
- Resumed Phase 6 after the scheduler cap cleared enough for a smoke:
  - `share/a40`, 1 GPU, 1 CPU, 16G, 20 minutes test-only predicted start on `cn-r-4`;
  - matrix smoke job `20480622` ran on `cn-r-4`, partition `share`, one `NVIDIA A40`, 1 CPU, 16G RAM;
  - job `20480622` completed with state `COMPLETED`, exit code `0:0`, elapsed `00:04:18`;
  - smoke log included `quat_mlp:True` and `quat_mlp_impl:matrix`.
- Matrix qMLP smoke facts:

```text
model_params:9982024
step:2/2 val_loss:6.9227 val_bpb:4.1000 train_time:615ms step_avg:307.63ms
peak memory allocated: 1748 MiB reserved: 1876 MiB
Serialized model int8+zlib: 6050224 bytes
Total submission size int8+zlib: 6101433 bytes
final_int8_zlib_roundtrip_exact val_loss:6.92820271 val_bpb:4.10327189
```

- Exact 2-CPU/24G matrix benchmark attempts:
  - job `20480631` was submitted after a favorable test-only result, stayed pending on `QOSGrpCpuLimit` with no start estimate, and was canceled;
  - job `20480651` was submitted after a later favorable test-only result, stayed pending on `QOSGrpCpuLimit` with no start estimate, and was canceled;
  - no exact-shape benchmark result exists yet.
- Provisional 1-CPU/16G matrix benchmark:
  - `share/a40`, 1 GPU, 1 CPU, 16G, 25 minutes test-only predicted start on `cn-r-6`;
  - job `20480636` ran on `cn-r-3`, partition `share`, one `NVIDIA A40`, 1 CPU, 16G RAM;
  - job `20480636` completed with state `COMPLETED`, exit code `0:0`, elapsed `00:15:07`;
  - this is not an exact replacement for Phase 3/5 because it used 1 CPU and 16G instead of 2 CPUs and 24G.
- Provisional matrix benchmark facts:

```text
model_params:9982024
train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:600.000
quat_mlp:True
quat_mlp_impl:matrix
step:368/20000 val_loss:2.7725 val_bpb:1.6420 train_time:600140ms step_avg:1630.81ms
stopping_early: wallclock_cap train_time:600140ms step:368/20000
peak memory allocated: 13047 MiB reserved: 13074 MiB
Serialized model: 38930363 bytes
Total submission size: 38981572 bytes
Serialized model int8+zlib: 8733639 bytes
Total submission size int8+zlib: 8784848 bytes
final_int8_zlib_roundtrip_exact val_loss:2.78364328 val_bpb:1.64863035
```

- Comparisons:

```text
dense_phase3_steps=379
split_qmlp_phase5_steps=263
matrix_qmlp_phase6_provisional_steps=368

dense_phase3_step_avg=1587.08ms
split_qmlp_phase5_step_avg=2283.03ms
matrix_qmlp_phase6_provisional_step_avg=1630.81ms

dense_phase3_roundtrip_val_bpb=1.58081095
split_qmlp_phase5_roundtrip_val_bpb=1.87232731
matrix_qmlp_phase6_provisional_roundtrip_val_bpb=1.64863035
```

- Matrix path deltas:
  - versus split qMLP, provisional matrix qMLP completed `105` more steps and improved roundtrip BPB by about `0.2237`;
  - versus dense baseline, provisional matrix qMLP completed `11` fewer steps, was about `2.8%` slower per step, and was worse by about `0.0678` BPB;
  - matrix qMLP artifact was `8733639` bytes, about `0.53 MB` smaller than dense baseline's `9265169` bytes, but BPB was worse.

Artifacts:

- Modified local `parameter-golf/train_gpt.py`.
- Updated local correctness script: `goal/4-qmlp-check.py`.
- Local Phase 6 plan: `goal/6-optimize.md`.
- Local smoke script: `goal/6-smoke.sbatch`.
- Local benchmark script: `goal/6-benchmark.sbatch`.
- Remote updated training script: `/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf/train_gpt.py`.
- Remote updated correctness script: `/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf/qmlp_check.py`.
- Remote staged smoke script: `/nfs/hpc/share/peterj29/pg/runs/phase6-matrix-smoke/phase6-smoke.sbatch`.
- Remote staged benchmark script: `/nfs/hpc/share/peterj29/pg/runs/phase6-matrix-benchmark/phase6-benchmark.sbatch`.
- Remote matrix smoke artifacts: `/nfs/hpc/share/peterj29/pg/runs/phase6-matrix-smoke/20480622/`.
- Remote provisional matrix benchmark artifacts: `/nfs/hpc/share/peterj29/pg/runs/phase6-matrix-benchmark/20480636/`.

New facts:

- Matrix qMLP is numerically equivalent to split qMLP for the tested Hamilton layout.
- Matrix qMLP preserves the same `9,982,024` model parameter count and `7,077,888` saved parameters.
- Matrix qMLP preserves the `72` qMLP 2D component matrices visible to Muon grouping.
- Matrix qMLP fixes most of the split implementation speed problem. The provisional benchmark was near dense speed and far faster than split qMLP.
- The remaining qMLP problem is BPB quality, not primarily throughput.
- Matrix qMLP's provisional run still trails the dense baseline by about `0.0678` roundtrip BPB at nearly the same step count.
- Current account/QOS pressure can block even a 1-CPU A40 smoke despite a favorable test-only result, so future submissions should be monitored and canceled if they remain stuck.

Decision:

- Proceed to a narrow saved-budget reinvestment phase, prioritizing vocabulary, because the matrix path fixed enough of the throughput problem to make the original question testable.
- Keep the exact 2-CPU/24G matrix rerun as a recommended follow-up when `QOSGrpCpuLimit` allows it, but do not loop on it before testing whether a larger tokenizer can buy back the BPB gap.
- Do not run a broad width/depth grid yet. If vocabulary reinvestment cannot beat dense, then run only a small qMLP quality phase such as init/LR before stopping the simple qMLP path.
- No Slurm jobs were left queued.
