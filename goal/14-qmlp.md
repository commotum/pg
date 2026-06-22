# Phase 14: Same-Vocab Record qMLP Tax

Date drafted: 2026-06-22

## Overview

Measure the cost of adding matrix qMLP to the same 04-23 record-stack configuration used in Phase 13, without changing vocabulary size or unrelated settings.

This phase uses CaseOps `sp8192`, the same record script target, the same data path, and the same A40 screening assumptions as Phase 13. The only intended model change is `QUAT_MLP=1` with `QUAT_MLP_IMPL=matrix`.

## Why This Matters

The simple-stack experiments showed that qMLP can save parameters and support useful vocabulary reinvestment, but they also showed that qMLP has an expressiveness/training tax and that larger vocab can hurt. Before testing qMLP `sp16384` in the record stack, we need to know the same-vocab qMLP tax inside the stronger stack.

This phase estimates:

```text
qMLP tax = record qMLP sp8192 BPB - record dense sp8192 BPB
```

Phase 15 can only be interpreted correctly after this tax is measured.

## Current Assumptions

- Phase 13 has produced or will produce a usable dense record `sp8192` A40 control.
- The chosen record target is:

```text
parameter-golf/records/track_10min_16mb/2026-04-23_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE
```

- A40 screening may use the local SDPA fallback when `flash_attn_interface` is unavailable.
- CaseOps `sp8192` data must exist before this phase runs:

```text
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
```

- The qMLP implementation has already been ported into the local 04-23 record `train_gpt.py`.

## Implementation Steps

1. Confirm Phase 13 produced a usable control.

Record:

- dense record `sp8192` job IDs;
- dense record `sp8192` BPB, steps, ms/step, package size, memory, hardware, and seed;
- any A40 compatibility switches, especially SDPA fallback.

2. Verify qMLP path and data prerequisites.

Check that:

- `goal/14-qmlp-smoke.sbatch` points to the same record stack and CaseOps `sp8192` data as Phase 13;
- `goal/14-qmlp-a40.sbatch` uses `VOCAB_SIZE=8192` implicitly through the tokenizer/data path;
- `QUAT_MLP=1` and `QUAT_MLP_IMPL=matrix` are set;
- train shards, validation shards, and validation byte sidecars exist.

3. Run the qMLP smoke.

Use `goal/14-qmlp-smoke.sbatch`.

The smoke should verify:

- import/init path;
- qMLP matrix path;
- A40 SDPA fallback if FA3 is unavailable;
- one or two training steps;
- package or prequant path;
- memory footprint.

4. Run A40 qMLP benchmarks.

Use `goal/14-qmlp-a40.sbatch`.

Prefer the same seeds as Phase 13. Run independent seeds in parallel when scheduler/account limits allow. If resource limits block parallel execution, run the maximum schedulable subset and document the fallback.

5. Compare against Phase 13.

For every seed, record:

- BPB delta versus dense record `sp8192`;
- steps and ms/step delta;
- package-size delta;
- memory delta;
- any changed failure modes.

## Expected Artifacts

- Slurm logs under `/nfs/hpc/share/peterj29/pg/runs/phase14-record-qmlp-*`.
- Per-job manifests, command logs, GPU logs, train logs, and package summaries where available.
- Updated `goal/14-qmlp.md` result block.
- Updated `goal/0-plan.md` and `goal/13-record.md` if new compatibility facts affect the record-stack control.

## Completion Requirements

This phase is complete when:

- a qMLP `sp8192` smoke reaches terminal state, or its blocker is captured;
- at least one qMLP `sp8192` A40 benchmark reaches terminal state if the smoke passes;
- the same-vocab qMLP tax is recorded against Phase 13 dense `sp8192`;
- job IDs, seeds, commands, BPB, steps, ms/step, package size, memory, and hardware are recorded;
- the decision says whether Phase 15 `sp16384` qMLP reinvestment should proceed.

## Failure and Fallback Rules

- If qMLP `sp8192` cannot initialize, fix the qMLP port before Phase 15.
- If qMLP `sp8192` is dramatically slower or much worse than dense `sp8192`, still record the tax; Phase 15 may be skipped if the tax is too large for vocabulary reinvestment to plausibly overcome.
- If A40 SDPA fallback is the only viable path, keep using it consistently across Phase 13, Phase 14, and Phase 15 A40 screening.
- Do not change tokenizer, optimizer policy, TTT policy, or compression settings unless the change is mechanically required and documented.

## Result

Status: pending

Evidence:

- Phase 14 scripts exist: `goal/14-qmlp-smoke.sbatch` and `goal/14-qmlp-a40.sbatch`.
- Phase 14 should not run until Phase 13 leaves a dense record `sp8192` control.

Decision:

- Pending Phase 13 completion.
