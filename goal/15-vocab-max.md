# Phase 15: Record-Stack qMLP Vocab-Max Search

Date drafted: 2026-06-22

## Overview

Find the largest CaseOps vocabulary that the record-stack qMLP variant can package under the 16 MB submission cap.

This phase is a package-size search, not a final BPB search. Use tiny qMLP package smokes to locate the vocab frontier, then run full A40 benchmarks only for the best safe candidate.

## Current Estimate

The same-vocab record qMLP parameter saving is exact:

```text
dense record MLP params = 23,068,672
qMLP record MLP params = 5,767,168
saved params = 17,301,504
```

Raw parameters would allow many more tokens, but the compressed artifact cap is tighter. The trained simple-stack slope from `sp4096 -> sp8192` was about `392` compressed submission bytes per added vocab token, so the expected frontier is probably around `11k-12k` unless the same-vocab record qMLP package saves much more than expected.

## Binary Search Procedure

Use these scripts with an explicit `VOCAB_SIZE`.

```text
VOCAB_SIZE=12288 sbatch goal/15-caseops-vocab.sbatch
VOCAB_SIZE=12288 sbatch goal/15-qmlp-package-smoke.sbatch
```

Read the package result from:

```text
/nfs/hpc/share/peterj29/pg/runs/phase15-qmlp-package-smoke/<job_id>/package-summary.txt
```

Decision rule:

- If total submission size is above `16,000,000`, lower the upper bound.
- If total submission size is below about `15,500,000`, raise the lower bound.
- If total submission size lands between about `15,500,000` and `16,000,000`, treat it as near-frontier and prefer one step lower for the full benchmark unless the later trained package has already proven safe.

Suggested first probes:

```text
12288
10240 or 14336 depending on the 12288 result
bisect in 512-token steps
stop at 256-token granularity only if the BPB curve later justifies it
```

## Completion Requirements

- CaseOps tokenizer/data exists for each probed vocab.
- A package smoke reaches terminal state for each probed vocab.
- The largest safe smoke vocab is recorded with total submission bytes.
- One full A40 qMLP record-stack benchmark is run for the selected safe vocab.
- If the trained package exceeds the cap or is too close to trust, rerun one lower vocab step.

## Notes

- Tiny package smokes can underpredict trained artifact size. Do not target an exact `15,999,999` byte package.
- Keep all record-stack settings fixed except `VOCAB_SIZE`, tokenizer/data paths, and `QUAT_MLP=1`.
- Do not use this phase to tune qMLP architecture or learning rates.
