# Phase 3: Parallel Package Smoke Matrix

Date drafted: 2026-06-23

## Overview

Run package/path smokes for the dense A40-friendly CaseOps model and the qMLP
variant across every CaseOps vocab that is currently ready.

This phase is non-sequential by design. It should submit ready cells now and add
new cells later as pending CaseOps exports finish.

## Why This Matters

The three-seed benchmark matrix should not spend A40 time on cells that cannot
find their tokenizer/data, fail qMLP initialization, or exceed the 16 MB package
cap. A short package smoke gives a cheap gate for each `(model_variant,
vocab_size)` pair.

## Current Assumptions

- Phase 2 harness exists and passed dense `sp8192` smoke job `20485638`.
- Future successful harness runs should report `complete_file=1` after the
  parser ordering fix.
- CaseOps `sp2048`, `sp8192`, and `sp16384` currently have enough shards for
  smoke cells.
- CaseOps `sp1024` and `sp4096` are being exported by another agent and should
  be added when they verify cleanly.
- Smokes use seed `42`; benchmark seeds are handled in Phase 4.

## Implementation Steps

1. Verify live CaseOps data status for target vocabs.
2. Sync updated `goal-2` docs/scripts to the HPC checkout.
3. Use `goal-2/3-submit-smoke-matrix.sh` from the HPC submit node.
4. Submit one independent one-A40 smoke job per ready cell:

```text
model variants: dense, qmlp
vocab sizes: ready subset of 1024, 2048, 4096, 8192, 16384
seed: 42
```

5. Do not submit cells whose data export is incomplete unless using an explicit
   Slurm `afterok` dependency on the matching data-export job.
6. Monitor each job to terminal Slurm state.
7. Parse and summarize `metrics.env` for each completed smoke.

## Expected Artifacts

- `goal-2/3-submit-smoke-matrix.sh`
- Smoke run root under:

```text
/nfs/hpc/share/peterj29/pg/runs/goal2-phase3-smokes/
```

Each successful smoke should write:

- `manifest.txt`
- `gpu.txt`
- `command.txt`
- `train.log`
- `work-files.txt`
- `metrics.env`
- `metrics.json`
- `metrics.tsv`
- `COMPLETE.txt`

## Completion Requirements

Phase 3 is complete when:

- every currently available vocab has dense and qMLP smoke outcomes, or a
  captured blocker;
- newly completed vocab exports can be added as follow-up matrix cells without
  rewriting the harness;
- at least `sp8192` dense and `sp8192` qMLP have complete package-smoke results;
- each completed cell records package size, post-quant no-TTT BPB, peak memory,
  job ID, host, data path, tokenizer path, and Slurm state.

## Failure and Fallback Rules

- If a cell fails because data is missing, do not retry until the export verifies.
- If a cell exceeds 16 MB, mark it over budget and do not send it to Phase 4.
- If qMLP fails but dense works for the same vocab, fix qMLP before benchmarks.
- If queue pressure prevents full parallelism, submit the maximum schedulable
  subset and record which cells remain.
- Do not use H100/H200 or full phased TTT in this phase.

## Result

Status: in progress

Evidence:

- Phase file created.
- Local syntax checks passed for:
  - `goal-2/2-a40-harness.sbatch`
  - `goal-2/2-parse-metrics.py`
  - `goal-2/3-submit-smoke-matrix.sh`
- Remote syntax checks passed after syncing to `/nfs/hpc/share/peterj29/pg/src/pg/goal-2/`.
- Dry-run at `2026-06-23T00:18 PDT` found ready cells for `sp2048`,
  `sp8192`, and `sp16384`, and skipped incomplete `sp1024` and `sp4096`.
- `sp1024` then reached 80 train shards and was submitted separately.

Submitted smoke jobs:

| Vocab | Model | Job ID | Latest State |
| --- | --- | ---: | --- |
| `1024` | dense | `20485707` | running on `cn-r-1` |
| `1024` | qMLP | `20485708` | completed |
| `2048` | dense | `20485698` | completed |
| `2048` | qMLP | `20485699` | completed |
| `4096` | dense | `20485825` | pending |
| `4096` | qMLP | `20485826` | pending |
| `8192` | dense | `20485700` | completed |
| `8192` | qMLP | `20485701` | completed |
| `16384` | dense | `20485702` | completed, over budget |
| `16384` | qMLP | `20485703` | completed |

Completed smoke metrics:

| Vocab | Model | Job ID | Quant BPB | Total Bytes | Peak MiB | Host | TTT |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `1024` | qMLP | `20485708` | `4.55731441` | `6507663` | `6771` | `cn-r-5` | `0` |
| `2048` | qMLP | `20485699` | `4.34588020` | `6777950` | `6805` | `cn-r-5` | `0` |
| `2048` | dense | `20485698` | `4.34837669` | `14277213` | `7062` | `cn-r-2` | `0` |
| `8192` | dense | `20485700` | `4.17584110` | `15912062` | `7756` | `cn-r-5` | `0` |
| `8192` | qMLP | `20485701` | `4.17319096` | `8421755` | `7497` | `cn-r-1` | `0` |
| `16384` | dense | `20485702` | `4.20013989` | `18106381` | `8683` | `cn-r-5` | `0` |
| `16384` | qMLP | `20485703` | `4.19746524` | `10615703` | `8423` | `cn-r-5` | `0` |

Skipped cells:

| Vocab | Reason |
| --- | --- |
| none | all target vocab exports have reached Phase 3 or later |

Artifacts:

- Launcher: `goal-2/3-submit-smoke-matrix.sh`
- Remote run root: `/nfs/hpc/share/peterj29/pg/runs/goal2-phase3-smokes/`

New facts:

- The Phase 3 launcher can be rerun incrementally without resubmitting already
  launched cells.
- The current smoke matrix covers all five target vocabs: `1024`, `2048`,
  `4096`, `8192`, and `16384`.
- The first two completed smoke cells are under the 16 MB cap and did not run
  TTT.
- qMLP `sp2048` and dense `sp8192` were released to Phase 4 three-seed
  benchmarks immediately after passing smoke, without waiting for the rest of
  Phase 3.
- qMLP `sp8192` and qMLP `sp16384` were also released to Phase 4 after passing
  smoke.
- Dense `sp2048` passed smoke under the 16 MB cap and was released to Phase 4.
- qMLP `sp1024` passed smoke under the 16 MB cap and was released to Phase 4.
- Dense `sp16384` passed functionally but exceeded the 16 MB cap at `18106381`
  bytes, so it is excluded from Phase 4 unless the user explicitly wants an
  over-budget diagnostic.
- `sp4096` export job `20485661` completed, verified 80 train shards plus
  validation bytes, and dense/qMLP smoke jobs `20485825` and `20485826` were
  submitted.

Decision:

- Monitor remaining Phase 3 jobs `20485707`, `20485825`, and `20485826`.
- Release dense `sp1024` and both `sp4096` variants to Phase 4 as soon as their
  individual smoke jobs pass under the 16 MB cap.
