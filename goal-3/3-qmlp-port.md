# Phase 3: qMLP Port

## Overview

Port the matrix qMLP implementation into the selected 2026-04-27 H100 record
stack while preserving the dense/base path when `QUAT_MLP=0`.

Staged source:

```text
goal-3/stage/primary-qmlp/train_gpt.py
```

Source base:

```text
parameter-golf/records/track_10min_16mb/
2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/train_gpt.py
```

## Implementation Steps

1. Copied the primary record stack into `goal-3/stage/primary-qmlp/`.
2. Added opt-in qMLP hyperparameters:
   - `QUAT_MLP`;
   - `QUAT_MLP_IMPL=matrix`.
3. Added Hamilton matrix helpers:
   - `_quaternion_matrix`;
   - `_quaternion_input_hessian`;
   - `_QUAT_COMPONENTS`;
   - `_is_quat_mlp_component`.
4. Updated `MLP.forward` so dense weights behave as before, while compact qMLP
   banks are materialized into Hamilton dense matrices before the existing fused
   LeakyReLU-square MLP path.
5. Updated `GPT.__init__` to allocate either dense MLP banks or compact qMLP
   banks:
   - dense up/down banks keep the original shapes;
   - qMLP up bank uses `(num_layers, 4, hidden_dim // 4, model_dim // 4)`;
   - qMLP down bank uses `(num_layers, 4, model_dim // 4, hidden_dim // 4)`.
6. Added early qMLP shape checks for `model_dim` and `hidden_dim` divisibility
   by 4.
7. Updated MLP bank initialization for qMLP component matrices.
8. Kept qMLP banks in the existing Muon matrix parameter group.
9. Updated GPTQ Hessian collection so qMLP component tensors receive compact
   quaternion-lane Hessians.
10. Forced qMLP component tensors through GPTQ instead of the small-tensor
    float16 passthrough path.
11. Updated per-group compression key ordering for qMLP component tensors.
12. Updated `_unbank_state_dict` and `_rebank_state_dict` for qMLP component
    names:
    - `blocks.{i}.mlp.fc.{wr,wi,wj,wk}`;
    - `blocks.{i}.mlp.proj.{wr,wi,wj,wk}`.
13. Updated deserialization to rebank qMLP tensors when `QUAT_MLP=1`.
14. Added run-log lines for `quat_mlp` and `quat_mlp_impl`.

## Verification

Local static verification completed:

```bash
python3 -m py_compile goal-3/stage/primary-qmlp/train_gpt.py
```

Result: passed.

Direct runtime verification is still pending because the script requires CUDA,
FA3/Triton, `sentencepiece`, and the CaseOps datasets. That verification belongs
in Slurm compute allocations, not on submit nodes.

## Findings

- The dense/base behavior remains opt-in by absence: `QUAT_MLP=0` keeps dense
  bank shapes and the original dense forward path.
- The qMLP path reuses the existing fused MLP kernel after Hamilton matrix
  materialization. This keeps the main MLP compute path consistent with the
  record stack but may add compile/runtime overhead that only a runtime smoke can
  measure.
- The 2026-04-27 per-group packer tolerates empty groups, so adding qMLP group
  keys should not break dense packages produced by the staged file.
- TTT LoRA hooks remain dimensionally unchanged because they act in model-space
  additive paths. Runtime smoke still needs to verify the compiled TTT path with
  qMLP enabled before final H100 use.

## Completion Requirements

- qMLP source port exists: complete.
- Dense path remains selectable: complete through `QUAT_MLP=0`.
- qMLP path remains selectable: complete through `QUAT_MLP=1`.
- Syntax compile passes: complete.
- qMLP runtime smoke: pending Phase 5 Slurm smoke.
- Package-size smoke: pending Phase 4.

## Next Phase

Phase 4 should create package and runtime-smoke scripts that can prove qMLP
serialization and under-16MB accounting before any H100 request.
