# Phase 4: Parallel Three-Seed A40 Matrix

Date drafted: 2026-06-23

## Overview

Run three A40 benchmark seeds for every dense/qMLP CaseOps vocab cell that
passes Phase 3 package smoke.

This phase uses the same lean harness as Phase 3, but switches to benchmark
mode and seeds `42`, `0`, and `1`.

## Why This Matters

Single-seed results are too noisy for deciding whether qMLP helps. The required
comparison is paired by vocab and seed:

```text
dense sp<VOCAB> seed S
qMLP  sp<VOCAB> seed S
```

The result should identify the best-under-16MB dense candidate and the
best-under-16MB qMLP candidate under the same A40-friendly no-TTT setup.

## Current Assumptions

- Phase 3 package smokes are the gate for Phase 4.
- Do not run benchmarks for cells with failed smoke, missing metrics, missing
  `COMPLETE.txt`, `ttt_seen=1`, or package size over 16,000,000 bytes.
- Use one A40 per job.
- Run independent cells in parallel where Slurm/account limits allow.
- If queue limits prevent full parallelism, submit the maximum schedulable subset
  and document pending cells.

## Implementation Steps

1. Wait for or poll Phase 3 smoke jobs to terminal state.
2. Summarize Phase 3 metrics.
3. Use `goal-2/4-submit-benchmark-matrix.sh` from the HPC submit node.
4. Dry-run first with `SUBMIT=0`; verify the selected cells and seeds.
5. Submit with `SUBMIT=1` only for smoke-passing cells.
6. Monitor all benchmark jobs to terminal state.
7. Parse `metrics.env` for every completed benchmark.
8. Compute per-vocab/per-model means and paired seed deltas.

## Expected Artifacts

- `goal-2/4-submit-benchmark-matrix.sh`
- Benchmark run root:

```text
/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/
```

## Completion Requirements

Phase 4 is complete when:

- every smoke-passing `(model_variant, vocab_size)` cell has seeds `42`, `0`,
  and `1`, or a documented blocker;
- every completed benchmark has parsed metrics;
- each metric row records vocab, model, seed, pre-quant BPB, post-quant no-TTT
  BPB, package size, train steps, speed, memory, host, and job ID;
- over-budget or failed cells are explicitly excluded from mean-based decisions.

## Failure and Fallback Rules

- If a benchmark cell fails because of a transient Slurm/node issue, rerun the
  same cell once.
- If a cell fails reproducibly because of model/data/package behavior, mark it
  failed and do not broaden the search.
- If a benchmark exceeds 16 MB despite passing smoke, mark it over budget.
- Do not add new seeds unless the user asks; the required seed set is `42`, `0`,
  and `1`.
- Do not use H100/H200 or full phased TTT in this phase.

## Result

Status: pending

Evidence:

- Phase file created.
- `goal-2/4-submit-benchmark-matrix.sh` was staged on HPC and syntax-checked.
- The launcher writes `/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/submitted.tsv` so it can be rerun incrementally without duplicating already-submitted cells.
- Per user direction, Phase 4 jobs are released for each individual smoke-passing setup as soon as it passes; the plan does not wait for all Phase 3 smokes to complete.

Released benchmark cells:

| Vocab | Model | Seed | Job ID | Smoke Gate |
| --- | --- | ---: | ---: | --- |
| `2048` | qMLP | `42` | `20485754` | smoke `20485699`, `6777950` bytes |
| `2048` | qMLP | `0` | `20485755` | smoke `20485699`, `6777950` bytes |
| `2048` | qMLP | `1` | `20485756` | smoke `20485699`, `6777950` bytes |
| `8192` | dense | `42` | `20485757` | smoke `20485700`, `15912062` bytes |
| `8192` | dense | `0` | `20485758` | smoke `20485700`, `15912062` bytes |
| `8192` | dense | `1` | `20485759` | smoke `20485700`, `15912062` bytes |
| `8192` | qMLP | `42` | `20485785` | smoke `20485701`, `8421755` bytes |
| `8192` | qMLP | `0` | `20485786` | smoke `20485701`, `8421755` bytes |
| `8192` | qMLP | `1` | `20485787` | smoke `20485701`, `8421755` bytes |
| `16384` | qMLP | `42` | `20485788` | smoke `20485703`, `10615703` bytes |
| `16384` | qMLP | `0` | `20485789` | smoke `20485703`, `10615703` bytes |
| `16384` | qMLP | `1` | `20485790` | smoke `20485703`, `10615703` bytes |

Artifacts:

- Launcher: `goal-2/4-submit-benchmark-matrix.sh`
- Remote run root: `/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/`
- Submission ledger: `/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/submitted.tsv`

New facts:

- Phase 4 can be safely advanced incrementally as each Phase 3 smoke cell passes.
- At the first release, qMLP `sp2048` and dense `sp8192` had passed smoke; their
  three-seed benchmark jobs are queued.
- At the second release, qMLP `sp8192` and qMLP `sp16384` had passed smoke; their
  three-seed benchmark jobs are queued.
- Dense `sp16384` passed smoke functionally, but the total submission size was
  `18106381` bytes, so it is excluded from compliant Phase 4 benchmarks.
- Dense/qMLP cells for `sp1024`, dense `sp2048`, and both `sp4096` variants
  should be released independently when their smokes pass and stay under the
  16 MB cap.

Decision:

- Continue monitoring Phase 3 smokes.
- Rerun the Phase 4 launcher with `SUBMIT=1` whenever additional smoke cells
  pass; the ledger should skip cells already submitted.
