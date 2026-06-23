# Phase 5: Matrix QA And Reruns

Date drafted: 2026-06-23

## Overview

Audit the Phase 4 benchmark matrix as jobs complete, repair only transient
failures, and summarize seed stability for dense versus qMLP at each CaseOps
vocab size.

This phase can run before all Phase 4 jobs finish. The QA script should be
rerun incrementally as new metrics land.

## Why This Matters

The goal is not a pile of independent Slurm jobs. The useful result is a
paired matrix that can answer whether qMLP improves BPB under the same lean A40
screening setup. Phase 5 turns raw per-job metrics into decision-grade rows:
means, standard deviations, paired dense-vs-qMLP deltas, and missing-cell
status.

## Current Assumptions

- Phase 4 benchmark jobs write `metrics.env` under:

```text
/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/<job_id>/
```

- The Phase 4 launcher writes `submitted.tsv` in the same run root.
- A valid compliant row has `complete_file=1`, `ttt_seen=0`, parsed BPB, and
  `total_submission_bytes <= 16000000`.
- User-approved over-budget diagnostics, currently dense `sp16384`, stay in the
  matrix but are excluded from best-under-cap decisions.

## Implementation Steps

1. Add a matrix summarizer that reads the Phase 4 run root.
2. Include submitted-but-missing metrics rows so pending cells are visible.
3. Compute per `(vocab_size, model_variant)`:
   - completed seed count;
   - seed list;
   - mean and sample standard deviation for quantized BPB;
   - mean and sample standard deviation for prequant BPB;
   - mean train steps;
   - over-budget completed count.
4. Compute paired seed deltas where dense and qMLP both completed for the same
   vocab and seed.
5. Write machine-readable TSV outputs and a Markdown summary.
6. Rerun the summarizer after each batch of new metrics.
7. Rerun Slurm cells only when the failure looks transient or when a paired
   comparison would otherwise be incomplete.

## Expected Artifacts

- `goal-2/5-summarize-matrix.py`
- Generated on HPC after the script runs:

```text
/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/matrix-summary/matrix-runs.tsv
/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/matrix-summary/matrix-summary.tsv
/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/matrix-summary/paired-deltas.tsv
/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/matrix-summary/matrix-summary.md
```

## Completion Requirements

Phase 5 is complete when:

- every Phase 4 submitted job is represented as complete, pending, failed,
  missing metrics, over budget, or blocked;
- every valid `(model_variant, vocab_size)` cell has three seeds or a documented
  blocker;
- each paired dense/qMLP comparison uses matching vocab and seed;
- the matrix summary identifies incomplete and over-budget rows explicitly;
- no rerun is launched without a specific reason.

## Failure And Fallback Rules

- If metrics parsing fails, inspect the run directory before rerunning the job.
- If a run is over budget, keep it as diagnostic only unless the user explicitly
  changes the cap.
- If a single seed fails due to node or scheduler failure, rerun that exact seed
  once.
- If a cell fails reproducibly because of model behavior, mark it failed and
  keep the rest of the matrix moving.
- Do not add extra seeds by default.

## Result

Status: in progress

Evidence:

- Phase file created.
- Matrix summarizer added at `goal-2/5-summarize-matrix.py`.
- Local syntax check passed with `python3 -m py_compile`.
- A no-data local smoke run wrote `/tmp/goal2-summary-test/matrix-summary.md`.
- The script was synced to HPC and ran successfully with the submit-node
  default `python3`.
- Current generated HPC summary:
  `/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/matrix-summary/matrix-summary.md`

Artifacts:

- `goal-2/5-summarize-matrix.py`
- `/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/matrix-summary/matrix-runs.tsv`
- `/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/matrix-summary/matrix-summary.tsv`
- `/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/matrix-summary/paired-deltas.tsv`
- `/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/matrix-summary/matrix-summary.md`

New facts:

- The first complete Phase 4 cell is qMLP `sp2048`, with seeds `42`, `0`, and
  `1` complete.
- Dense `sp8192` has two completed seeds so far: `42` and `1`.
- Dense `sp8192` now has all three seeds complete.
- qMLP `sp8192` has two completed seeds so far: `42` and `0`.
- The refreshed summarizer produced the first paired dense-vs-qMLP deltas:
  qMLP `sp8192` beat dense `sp8192` by `0.60712108` BPB on seed `42` and
  `0.67822520` BPB on seed `0`.

Decision:

- Run the summarizer on HPC after syncing and after each meaningful batch of new
  Phase 4 metrics.
