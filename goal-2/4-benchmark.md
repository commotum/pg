# Phase 4: Parallel Three-Seed A40 Matrix

Date drafted: 2026-06-23

## Overview

Run three A40 benchmark seeds for every dense/qMLP CaseOps vocab cell that
passes Phase 3 package smoke.

This phase uses the same lean harness as Phase 3, but switches to benchmark
mode and seeds `42`, `0`, and `1`.

The benchmark wallclock cap is a training-loop cap, not an end-to-end Slurm job
cap. The harness sets `MAX_WALLCLOCK_SECONDS=600` for training, then still has
to run validation, EMA, serialization, GPTQ calibration/quantization, brotli
packaging, and metrics parsing. A job can therefore show more than 10 minutes of
Slurm elapsed time while still respecting the 10-minute training cap.

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
- Do not run compliant benchmarks for cells with failed smoke, missing metrics,
  missing `COMPLETE.txt`, `ttt_seen=1`, or package size over 16,000,000 bytes.
- Explicit user-approved over-budget diagnostics may be submitted with
  `ALLOW_OVER_BUDGET=1`, but they must stay out of best-under-cap rankings.
- Use one A40 per job.
- Run independent cells in parallel where Slurm/account limits allow.
- Interpret `train_steps` and BPB as the fair screening metrics under the
  training cap. Interpret total Slurm elapsed as harness overhead and queue
  occupancy, not as extra training time.
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
- The launcher defaults to enforcing the 16 MB cap. `ALLOW_OVER_BUDGET=1` is
  available only for explicit diagnostics.
- The Phase 4 benchmark uses a 600-second training-loop cap, not a 600-second
  end-to-end Slurm cap. Job `20485758` confirmed this distinction: training
  stopped at `stopping_early: wallclock_cap train_time: 606200ms step: 64`, then
  continued through eval, serialization, GPTQ, and compression.
- Per user direction, Phase 4 jobs are released for each individual smoke-passing setup as soon as it passes; the plan does not wait for all Phase 3 smokes to complete.

Released benchmark and diagnostic cells:

| Vocab | Model | Seed | Job ID | Smoke Gate |
| --- | --- | ---: | ---: | --- |
| `1024` | dense | `42` | `20485866` | smoke `20485707`, `14009458` bytes |
| `1024` | dense | `0` | `20485867` | smoke `20485707`, `14009458` bytes |
| `1024` | dense | `1` | `20485868` | smoke `20485707`, `14009458` bytes |
| `1024` | qMLP | `42` | `20485820` | smoke `20485708`, `6507663` bytes |
| `1024` | qMLP | `0` | `20485821` | smoke `20485708`, `6507663` bytes |
| `1024` | qMLP | `1` | `20485822` | smoke `20485708`, `6507663` bytes |
| `2048` | dense | `42` | `20485814` | smoke `20485698`, `14277213` bytes |
| `2048` | dense | `0` | `20485815` | smoke `20485698`, `14277213` bytes |
| `2048` | dense | `1` | `20485816` | smoke `20485698`, `14277213` bytes |
| `2048` | qMLP | `42` | `20485754` | smoke `20485699`, `6777950` bytes |
| `2048` | qMLP | `0` | `20485755` | smoke `20485699`, `6777950` bytes |
| `2048` | qMLP | `1` | `20485756` | smoke `20485699`, `6777950` bytes |
| `4096` | dense | `42` | `20486127` | smoke `20485825`, `14818388` bytes |
| `4096` | dense | `0` | `20486128` | smoke `20485825`, `14818388` bytes |
| `4096` | dense | `1` | `20486129` | smoke `20485825`, `14818388` bytes |
| `4096` | qMLP | `42` | `20486109` | smoke `20485826`, `7324881` bytes |
| `4096` | qMLP | `0` | `20486110` | smoke `20485826`, `7324881` bytes |
| `4096` | qMLP | `1` | `20486111` | smoke `20485826`, `7324881` bytes |
| `8192` | dense | `42` | `20485757` | smoke `20485700`, `15912062` bytes |
| `8192` | dense | `0` | `20485758` | smoke `20485700`, `15912062` bytes |
| `8192` | dense | `1` | `20485759` | smoke `20485700`, `15912062` bytes |
| `8192` | qMLP | `42` | `20485785` | smoke `20485701`, `8421755` bytes |
| `8192` | qMLP | `0` | `20485786` | smoke `20485701`, `8421755` bytes |
| `8192` | qMLP | `1` | `20485787` | smoke `20485701`, `8421755` bytes |
| `16384` | dense | `42` | `20485837` | smoke `20485702`, `18106381` bytes, over budget |
| `16384` | dense | `0` | `20485838` | smoke `20485702`, `18106381` bytes, over budget |
| `16384` | dense | `1` | `20485839` | smoke `20485702`, `18106381` bytes, over budget |
| `16384` | qMLP | `42` | `20485788` | smoke `20485703`, `10615703` bytes |
| `16384` | qMLP | `0` | `20485789` | smoke `20485703`, `10615703` bytes |
| `16384` | qMLP | `1` | `20485790` | smoke `20485703`, `10615703` bytes |

Completed benchmark metrics:

| Vocab | Model | Seed | Job ID | Quant BPB | Prequant BPB | Train Steps | Total Bytes | Peak MiB | Host | TTT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `1024` | dense | `42` | `20485866` | `4.62377075` | `4.59578191` | `66` | `14037902` | `26402` | `cn-r-4` | `0` |
| `1024` | dense | `0` | `20485867` | `4.65412191` | `4.62872343` | `66` | `14041173` | `26402` | `cn-r-4` | `0` |
| `1024` | dense | `1` | `20485868` | `4.58583266` | `4.56014747` | `66` | `14040047` | `26402` | `cn-r-1` | `0` |
| `1024` | qMLP | `42` | `20485820` | `3.35503787` | `3.35051733` | `66` | `6539552` | `26077` | `cn-r-2` | `0` |
| `1024` | qMLP | `0` | `20485821` | `3.36917824` | `3.36393715` | `66` | `6537052` | `26077` | `cn-r-4` | `0` |
| `1024` | qMLP | `1` | `20485822` | `3.36968348` | `3.36458550` | `66` | `6536757` | `26077` | `cn-r-5` | `0` |
| `2048` | dense | `42` | `20485814` | `4.28219257` | `4.26377910` | `66` | `14317100` | `26537` | `cn-r-5` | `0` |
| `2048` | dense | `0` | `20485815` | `4.23759240` | `4.22273113` | `65` | `14316662` | `26537` | `cn-r-5` | `0` |
| `2048` | dense | `1` | `20485816` | `4.27362994` | `4.25696812` | `66` | `14317323` | `26537` | `cn-r-1` | `0` |
| `2048` | qMLP | `42` | `20485754` | `3.27018661` | `3.26684355` | `66` | `6812458` | `26212` | `cn-r-5` | `0` |
| `2048` | qMLP | `0` | `20485755` | `3.27763883` | `3.27243947` | `66` | `6814438` | `26212` | `cn-r-2` | `0` |
| `2048` | qMLP | `1` | `20485756` | `3.22433420` | `3.21899798` | `66` | `6815170` | `26212` | `cn-r-5` | `0` |
| `4096` | qMLP | `42` | `20486109` | `3.11298502` | `3.10981282` | `65` | `7366939` | `27119` | `cn-r-5` | `0` |
| `4096` | qMLP | `0` | `20486110` | `3.12329786` | `3.12075511` | `65` | `7367399` | `27119` | `cn-r-5` | `0` |
| `8192` | dense | `42` | `20485757` | `3.63584015` | `3.62495393` | `64` | `15948267` | `29258` | `cn-r-5` | `0` |
| `8192` | dense | `0` | `20485758` | `3.68862843` | `3.67814286` | `64` | `15946672` | `29258` | `cn-r-1` | `0` |
| `8192` | dense | `1` | `20485759` | `3.65672897` | `3.64769098` | `64` | `15946703` | `29258` | `cn-r-5` | `0` |
| `8192` | qMLP | `42` | `20485785` | `3.02871907` | `3.02677027` | `64` | `8464901` | `28931` | `cn-r-2` | `0` |
| `8192` | qMLP | `0` | `20485786` | `3.01040323` | `3.00791804` | `64` | `8464608` | `28931` | `cn-r-5` | `0` |
| `8192` | qMLP | `1` | `20485787` | `3.01325051` | `3.01171267` | `64` | `8465289` | `28931` | `cn-r-5` | `0` |
| `16384` | dense | `42` | `20485837` | `3.60139160` | `3.59443636` | `61` | `18133059` | `32891` | `cn-r-5` | `0` |
| `16384` | dense | `0` | `20485838` | `3.56315533` | `3.55324698` | `62` | `18134380` | `32891` | `cn-r-5` | `0` |
| `16384` | dense | `1` | `20485839` | `3.57480857` | `3.57024485` | `62` | `18132809` | `32891` | `cn-r-2` | `0` |
| `16384` | qMLP | `42` | `20485788` | `2.99116918` | `2.99016518` | `62` | `10654489` | `32564` | `cn-r-1` | `0` |
| `16384` | qMLP | `0` | `20485789` | `2.99607641` | `2.99378620` | `62` | `10654853` | `32564` | `cn-r-2` | `0` |
| `16384` | qMLP | `1` | `20485790` | `3.00104931` | `2.99901433` | `62` | `10655635` | `32564` | `cn-r-5` | `0` |

Completed cell summaries:

| Vocab | Model | Seeds | Mean Quant BPB | Mean Prequant BPB | Mean Steps | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `1024` | dense | `42,0,1` | `4.62124177` | `4.59488427` | `66` | complete under-cap dense control; qMLP wins matched mean by `1.25660857` BPB |
| `1024` | qMLP | `42,0,1` | `3.36463320` | `3.35967999` | `66` | complete under-cap qMLP cell |
| `2048` | dense | `42,0,1` | `4.26447164` | `4.24782612` | `65.67` | complete under-cap dense control; qMLP wins matched mean by `1.00708509` BPB |
| `2048` | qMLP | `42,0,1` | `3.25738655` | `3.25276033` | `66` | first complete Phase 4 cell |
| `8192` | dense | `42,0,1` | `3.66039918` | `3.65026259` | `64` | complete under-cap dense baseline cell |
| `8192` | qMLP | `42,0,1` | `3.01745760` | `3.01546700` | `64` | complete under-cap qMLP cell |
| `16384` | dense | `42,0,1` | `3.57978517` | `3.57264240` | `61.67` | complete over-budget diagnostic; all three runs exceed 16 MB |
| `16384` | qMLP | `42,0,1` | `2.99609830` | `2.99432190` | `62` | current best qMLP cell |

Partial cell summaries:

| Vocab | Model | Seeds | Mean Quant BPB | Mean Prequant BPB | Mean Steps | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `4096` | qMLP | `42,0` | `3.11814144` | `3.11528397` | `65` | seed `1` still running/missing metrics |

Current paired deltas:

| Vocab | Seed | Dense Quant BPB | qMLP Quant BPB | qMLP - Dense |
| --- | ---: | ---: | ---: | ---: |
| `1024` | `42` | `4.62377075` | `3.35503787` | `-1.26873288` |
| `1024` | `0` | `4.65412191` | `3.36917824` | `-1.28494367` |
| `1024` | `1` | `4.58583266` | `3.36968348` | `-1.21614918` |
| `2048` | `42` | `4.28219257` | `3.27018661` | `-1.01200596` |
| `2048` | `0` | `4.23759240` | `3.27763883` | `-0.95995357` |
| `2048` | `1` | `4.27362994` | `3.22433420` | `-1.04929574` |
| `8192` | `42` | `3.63584015` | `3.02871907` | `-0.60712108` |
| `8192` | `0` | `3.68862843` | `3.01040323` | `-0.67822520` |
| `8192` | `1` | `3.65672897` | `3.01325051` | `-0.64347846` |

Diagnostic over-budget paired deltas:

| Vocab | Seed | Dense Quant BPB | qMLP Quant BPB | qMLP - Dense | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `16384` | `42` | `3.60139160` | `2.99116918` | `-0.61022242` | dense package is over cap at `18133059` bytes |
| `16384` | `0` | `3.56315533` | `2.99607641` | `-0.56707892` | dense package is over cap at `18134380` bytes |
| `16384` | `1` | `3.57480857` | `3.00104931` | `-0.57375926` | dense package is over cap at `18132809` bytes |

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
- At the third release, dense `sp2048` had passed smoke; its three-seed
  benchmark jobs are queued.
- At the fourth release, qMLP `sp1024` had passed smoke; its three-seed
  benchmark jobs are queued.
- At the fifth compliant release, dense `sp1024` had passed smoke; its
  three-seed benchmark jobs are queued.
- Dense `sp16384` passed smoke functionally, but the total submission size was
  `18106381` bytes, so it is excluded from compliant Phase 4 benchmarks.
- Per user direction, dense `sp16384` was later released as an over-budget
  diagnostic control to compare against qMLP `sp16384`. It remains excluded from
  compliant best-under-16MB rankings.
- qMLP `sp2048` seeds `42`, `0`, and `1` completed. This is the first complete
  three-seed Phase 4 cell.
- Dense `sp8192` seeds `42`, `0`, and `1` completed under the 16 MB cap. This
  is the first complete dense Phase 4 cell.
- qMLP `sp8192` seeds `42`, `0`, and `1` completed under the 16 MB cap. The
  three-seed mean beats dense `sp8192` by `0.64294158` BPB.
- qMLP `sp16384` seeds `42`, `0`, and `1` completed under the 16 MB cap with
  mean quantized BPB `2.99609830`.
- Dense `sp2048` seed `42` completed under the 16 MB cap. On the paired seed,
  qMLP `sp2048` beats it by `1.01200596` BPB.
- Dense `sp2048` seed `0` completed under the 16 MB cap. On the paired seed,
  qMLP `sp2048` beats it by `0.95995357` BPB.
- Dense `sp2048` seed `1` completed under the 16 MB cap. Dense `sp2048` is now
  a complete three-seed cell with mean quantized BPB `4.26447164`; qMLP
  `sp2048` wins the matched mean by `1.00708509` BPB.
- qMLP `sp4096` smoke passed under the 16 MB cap and released three Phase 4
  benchmark jobs: `20486109`, `20486110`, and `20486111`.
- qMLP `sp1024` seeds `42`, `0`, and `1` completed under the 16 MB cap with
  mean quantized BPB `3.36463320`.
- Dense `sp1024` seeds `42`, `0`, and `1` completed under the 16 MB cap with
  mean quantized BPB `4.62124177`; qMLP `sp1024` wins the matched mean by
  `1.25660857` BPB.
- qMLP `sp4096` seeds `42` and `0` completed under the 16 MB cap. The partial
  two-seed mean quantized BPB is `3.11814144`; seed `1` is still pending
  metrics.
- Dense `sp4096` smoke job `20485825` completed under the 16 MB cap at
  `14818388` total submission bytes and released three Phase 4 benchmark jobs:
  `20486127`, `20486128`, and `20486129`.
- Dense diagnostic `sp16384` seed `42` completed over budget with quantized BPB
  `3.60139160` and total submission size `18133059` bytes. It is useful as an
  over-budget control only and remains excluded from compliant rankings.
- Dense diagnostic `sp16384` seed `0` completed over budget with quantized BPB
  `3.56315533` and total submission size `18134380` bytes. It remains excluded
  from compliant rankings.
- Dense diagnostic `sp16384` seed `1` completed over budget with quantized BPB
  `3.57480857` and total submission size `18132809` bytes. Dense `sp16384` now
  has all three diagnostic seeds complete with mean quantized BPB `3.57978517`;
  all three runs remain excluded from compliant rankings.
- User direction added `sp32768` to the goal for both dense and qMLP formats
  with three seeds each. These cells must enter Phase 4 only after the
  `sp32768` CaseOps export verifies and dense/qMLP Phase 3 smokes complete.
- Dense/qMLP `sp32768` smoke jobs are queued behind the export dependency as
  jobs `20486178` and `20486179`; do not submit `sp32768` benchmarks until
  those smokes produce metrics.
- A fresh Phase 4 launcher dry-run confirmed that all smoke-passing cells already
  have submitted benchmark seeds in the ledger, dense `sp16384` remains skipped
  by default as over budget, and `sp32768` dense/qMLP are skipped because Phase
  3 smoke metrics do not exist yet.

Decision:

- Continue monitoring Phase 4 jobs until all submitted cells have either
  completed, failed, or been explicitly excluded.
- Rerun the Phase 5 summarizer after each meaningful batch of new metrics.
- After `sp32768` smokes complete, submit three benchmark seeds for dense and
  qMLP. If dense or qMLP `sp32768` exceeds 16 MB, keep it labeled as a
  user-directed diagnostic control.
