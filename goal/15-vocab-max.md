# Phase 15: Record-Stack qMLP sp16384 Reinvestment

Date drafted: 2026-06-22

## Overview

Test a fixed CaseOps `VOCAB_SIZE=16384` record-stack qMLP candidate on A40.

Canonical execution file: `goal/15-sp16384.md`. This file preserves the revised package-frontier context and deferred search machinery.

This replaces the earlier immediate vocab-frontier search. The `11776` package canary showed the record-stack qMLP package path has much more headroom than expected, but the next scientific question is simpler than "how large can vocab get":

```text
Does record-stack qMLP sp16384 beat the record-stack controls at sp8192?
```

The frontier search remains useful later, but it is deferred until after the three-way A40 comparison exists.

## Current Package Evidence

The same-vocab record qMLP parameter saving is exact:

```text
dense record MLP params = 23,068,672
qMLP record MLP params = 5,767,168
saved params = 17,301,504
```

Raw parameters would allow many more tokens, but the compressed artifact cap is tighter. Before the record-stack canary, the trained simple-stack slope from `sp4096 -> sp8192` suggested a rough frontier around `11k-12k`.

That old `11k-12k` frontier estimate is superseded by the Phase 15 canary:

```text
job_id=20484777
vocab_size=11776
state=COMPLETED
quantized_model_brotli_bytes=9344417
total_submission_bytes=9376232
```

The canary makes `16384` safe enough to use as the fixed larger-vocab benchmark point. It does not prove that trained `16384` is final-package safe under all conditions, so still run a tiny package/path smoke before full A40 benchmarks.

## Deferred Frontier Procedure

Do not run this procedure before the fixed `sp16384` benchmark unless the user explicitly redirects the phase.

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

Deferred probes:

```text
24576
32768 if 24576 has enough headroom
then binary search in 512-token steps if frontier probing becomes relevant again
```

## Active Phase Procedure

1. Ensure Phase 13 dense record `sp8192` and Phase 14 qMLP record `sp8192` controls exist.
2. Build or stage CaseOps `VOCAB_SIZE=16384` tokenizer/data for the 04-23 record path.
3. Run a qMLP `sp16384` package/path smoke. This is a guard for wiring, memory, and package-size surprises, not an open-ended search gate.
4. If the smoke is mechanically valid and not unexpectedly near or over 16 MB, run A40 qMLP `sp16384` benchmarks with the same seed policy as Phases 13 and 14.
5. Compare:

```text
dense record sp8192
qMLP record sp8192
qMLP record sp16384
```

## Completion Requirements

- CaseOps tokenizer/data exists for `VOCAB_SIZE=16384`.
- A qMLP `sp16384` package/path smoke reaches terminal state.
- At least one full A40 qMLP record-stack `sp16384` benchmark reaches terminal state if the smoke passes.
- Results are compared against dense record `sp8192` and qMLP record `sp8192`.
- The decision says whether larger-vocab frontier probing should reopen or remain deferred.

## Notes

- Tiny package smokes can underpredict trained artifact size. Do not target an exact `15,999,999` byte package.
- Keep all record-stack settings fixed except `VOCAB_SIZE`, tokenizer/data paths, and `QUAT_MLP=1`.
- Do not use this phase to tune qMLP architecture or learning rates.
- If qMLP `sp16384` loses to qMLP `sp8192`, treat that as evidence that the simple-stack `sp16384` failure transfers to the record stack and stop larger-vocab probing unless a new reason appears.
