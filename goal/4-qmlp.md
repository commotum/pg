# Phase 4: Quaternion MLP Correctness

Date drafted: 2026-06-22

## Overview

Add a feature-flagged quaternion MLP path to `parameter-golf/train_gpt.py` and prove that it works correctly before any A40 benchmark comparison.

The Phase 4 target is not lower BPB yet. It is correctness: shape compatibility, gradient flow, optimizer inclusion, parameter-count reduction, `torch.compile` compatibility, and a short end-to-end training smoke.

## Why This Matters

The qMLP idea can only be evaluated fairly if the implementation is mechanically sound. A silent optimizer omission, wrong Hamilton signs, broken compile path, or unintended attention/tokenizer change would make any benchmark result misleading.

This phase moves us closer to the qMLP decision by creating the smallest safe implementation that can be compared against the Phase 3 baseline.

## Assumptions and Dependencies

- Phase 3 A40 baseline benchmark is complete.
- Baseline reference:
  - `model_params:17059912`;
  - `379` steps in about 600 seconds;
  - `step_avg:1587.08ms`;
  - final roundtrip `val_bpb:1.58081095`.
- Initial qMLP target is only the MLP `fc` and `proj` projections.
- Attention, tokenizer, optimizer type, data loader, quantization, and validation logic should remain unchanged.
- `train_gpt.py` sends 2D block parameters to Muon, so quaternion component weights should be separate 2D `nn.Parameter` values.
- `MODEL_DIM` and `MLP_MULT * MODEL_DIM` must be divisible by 4 when `QUAT_MLP=1`.

## Implementation Steps

1. Add `quat_mlp` to `Hyperparameters`.

```python
quat_mlp = bool(int(os.environ.get("QUAT_MLP", "0")))
```

2. Add a `QuaternionLinear` module.

Requirements:

- no bias for the initial implementation;
- separate 2D parameters `wr`, `wi`, `wj`, `wk`;
- each component parameter shape `(out_features // 4, in_features // 4)`;
- input and output dimensions divisible by 4;
- forward path splits the last dimension into four components;
- applies Hamilton multiplication signs explicitly;
- concatenates four output components along the last dimension;
- casts weights to the input dtype in the same spirit as `CastedLinear`;
- supports `_zero_init = True` for MLP projection zero initialization.

3. Update `MLP`.

```python
self.fc = linear_cls(dim, hidden, bias=False)
self.proj = linear_cls(hidden, dim, bias=False)
```

Use `QuaternionLinear` only when `QUAT_MLP=1`.

4. Thread the feature flag through:

- `GPT.__init__`;
- `Block.__init__`;
- `MLP.__init__`;
- model construction in `main`.

5. Update initialization.

The existing `_init_weights` loop initializes zeroed `nn.Linear` modules. It must also zero any `QuaternionLinear` with `_zero_init = True`.

6. Add logging.

Log at least:

```text
quat_mlp:<True|False>
```

If useful, also log the matrix/scalar optimizer parameter counts.

7. Add local correctness checks.

Use a small local test script or `python -c` snippets to verify:

- `QuaternionLinear(512, 1024)` output shape;
- `QuaternionLinear(1024, 512)` output shape;
- all four component weights receive gradients;
- parameter count for qMLP baseline matches expected reduction;
- no qMLP block matrix parameter is excluded from Muon grouping;
- `GPT(... quat_mlp=True)` can run a toy forward/backward on CPU.

8. Run a Slurm qMLP smoke job.

Use the same small smoke shape as Phase 2, but with `QUAT_MLP=1`:

```text
RUN_ID=qmlp_smoke_a40_job<job-id>
QUAT_MLP=1
DATA_PATH=/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf/data/datasets/fineweb10B_sp1024
TOKENIZER_PATH=/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
VOCAB_SIZE=1024
SEED=42
ITERATIONS=2
WARMUP_STEPS=1
TRAIN_LOG_EVERY=1
VAL_LOSS_EVERY=0
TRAIN_BATCH_TOKENS=65536
VAL_BATCH_SIZE=65536
MAX_WALLCLOCK_SECONDS=0
```

Use A40, not RTX8000, because the baseline script depends on bf16/flash CUDA paths.

## Expected Artifacts

- Modified `parameter-golf/train_gpt.py`.
- Local correctness command outputs recorded in this file.
- Local qMLP smoke batch script if created.
- Remote qMLP smoke artifacts under `/nfs/hpc/share/peterj29/pg/runs/phase4-qmlp/`.
- Updated `goal/0-plan.md`.
- Updated this phase file.

## Completion Requirements

This phase is complete when:

- `QUAT_MLP=0` preserves the baseline code path.
- `QUAT_MLP=1` replaces only MLP `fc` and `proj` with `QuaternionLinear`.
- Local correctness checks pass.
- qMLP parameter count reduction is recorded.
- Muon optimizer receives qMLP component matrices.
- A qMLP Slurm smoke job reaches state `COMPLETED`.
- qMLP smoke exit code is `0:0`.
- qMLP smoke log includes `model_params`, `quat_mlp:True`, at least one completed training iteration, final `val_bpb`, and int8+zlib roundtrip metrics.
- qMLP smoke writes `final_model.int8.ptz`.
- The result section below records evidence, artifacts, new facts, and the Phase 5 decision.

## Failure and Fallback Rules

- If the local CPU toy forward fails, fix that before running Slurm.
- If `torch.compile` fails on qMLP, capture the exact error and try a minimal expression rewrite before changing model scope.
- If qMLP smoke OOMs, reduce smoke batch sizes before changing qMLP design.
- If qMLP is very slow, still complete the correctness smoke if it fits the short walltime; speed is Phase 5's concern.
- If optimizer grouping misses any qMLP 2D matrix, fix grouping before training.
- Do not change attention, tokenizer, validation, or quantization in this phase.

## Result

Status: complete as of 2026-06-22 02:02 PDT.

Evidence:

- `parameter-golf/train_gpt.py` now has a feature-flagged `QUAT_MLP=1` path.
- `QUAT_MLP=0` keeps the dense `CastedLinear` MLP path.
- `QUAT_MLP=1` replaces only `MLP.fc` and `MLP.proj` with `QuaternionLinear`; attention, tokenizer, validation, data loading, and quantized roundtrip logic are unchanged.
- `QuaternionLinear` stores `wr`, `wi`, `wj`, and `wk` as separate 2D parameters, which keeps them eligible for the existing Muon `p.ndim == 2` matrix grouping.
- Local syntax check passed with `python3 -m py_compile parameter-golf/train_gpt.py`.
- Remote CPU correctness check passed:

```text
shape_checks=ok
hamilton_equivalence_max_error=1.19e-07
dense_params=17059912
qmlp_params=9982024
saved_params=7077888
qmlp_muon_matrix_params=72
toy_forward_backward=ok loss=3.475729
```

- A40 qMLP smoke job `20480598` completed on `cn-r-4`, partition `share`, one `NVIDIA A40`, 1 CPU, 16G RAM, and 20 minute walltime.
- Job `20480598` completed with Slurm state `COMPLETED`, exit code `0:0`, elapsed `00:07:18`.
- The first 2-CPU qMLP smoke request, job `20480593`, was canceled while pending on `QOSGrpCpuLimit`. A 1-CPU / 16G A40 test-only request was schedulable immediately and was enough for the smoke.
- qMLP smoke config:

```text
RUN_ID=qmlp_smoke_a40_job20480598
QUAT_MLP=1
VOCAB_SIZE=1024
SEED=42
ITERATIONS=2
WARMUP_STEPS=1
TRAIN_LOG_EVERY=1
VAL_LOSS_EVERY=0
TRAIN_BATCH_TOKENS=65536
VAL_BATCH_SIZE=65536
MAX_WALLCLOCK_SECONDS=0
```

- qMLP smoke log facts:

```text
model_params:9982024
quat_mlp:True
world_size:1 grad_accum_steps:8
step:1/2 train_loss:6.9365 train_time:411ms step_avg:410.93ms
step:2/2 train_loss:6.9313 train_time:802ms step_avg:401.17ms
step:2/2 val_loss:6.9227 val_bpb:4.1000 train_time:803ms step_avg:401.60ms
peak memory allocated: 1796 MiB reserved: 1834 MiB
Serialized model: 38930363 bytes
Total submission size: 38980568 bytes
Serialized model int8+zlib: 6050417 bytes
Total submission size int8+zlib: 6100622 bytes
final_int8_zlib_roundtrip_exact val_loss:6.92820268 val_bpb:4.10327187
```

Artifacts:

- Local phase plan: `goal/4-qmlp.md`.
- Local correctness script: `goal/4-qmlp-check.py`.
- Local qMLP smoke script: `goal/4-qmlp.sbatch`.
- Remote qMLP correctness script: `/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf/qmlp_check.py`.
- Remote qMLP smoke artifacts: `/nfs/hpc/share/peterj29/pg/runs/phase4-qmlp/20480598/`.
- Remote smoke files:
  - `train.log`;
  - `manifest.txt`;
  - `gpu.txt`;
  - `python.txt`;
  - `work-files.txt`;
  - `work/final_model.pt`;
  - `work/final_model.int8.ptz`;
  - `work/logs/qmlp_smoke_a40_job20480598.txt`.

New facts:

- qMLP reduced the baseline model from `17,059,912` to `9,982,024` parameters, saving `7,077,888` parameters.
- qMLP has `72` 2D component matrices in the toy/default model, matching 9 layers * 2 MLP projections * 4 quaternion components.
- The qMLP smoke was slower than the dense Phase 2 smoke at the same tiny 2-step shape: about `401.60ms/step` qMLP versus about `262ms/step` dense. This speed penalty is plausible because the current qMLP forward uses several `F.linear` calls per quaternion projection.
- qMLP smoke int8+zlib artifact was `6,050,417` bytes versus dense Phase 2 smoke artifact `4,963,374` bytes at the same 2-step smoke shape. Lower parameter count does not guarantee smaller compressed artifact after brief training.
- qMLP final roundtrip smoke BPB was `4.10327187`, essentially parity with the dense Phase 2 smoke roundtrip `4.10409907`; the 2-step smoke is a correctness check, not a quality decision.
- Remote `parameter-golf` is intentionally a modified working copy based on commit `f5c079314c4877fbb0af378c0abade5a8ca33d3a`; manifest status recorded ` M train_gpt.py;?? qmlp_check.py;`.

Decision:

- Proceed to Phase 5: run a same-wallclock A40 benchmark with `QUAT_MLP=1`.
- Phase 5 must treat speed as a first-class result. If qMLP completes far fewer steps or worsens BPB at the same 600-second budget, optimize the implementation before spending saved parameters on vocabulary, width, or depth.
- The initial qMLP implementation is correct enough to benchmark, but not yet proven useful for Parameter Golf because BPB, not parameter count, is the goal.
