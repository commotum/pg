# Phase 1: CaseOps Data Matrix

Date drafted: 2026-06-22

## Overview

Prepare or verify the CaseOps tokenizer/data exports required by the lean A40
qMLP vocab ladder.

Target vocab sizes:

```text
1024
2048
4096
8192
16384
```

## Why This Matters

The Phase 2 harness selects CaseOps data by `VOCAB_SIZE`. Dense `sp8192` and
qMLP `sp8192` can run from the existing patched export, and `sp16384` can run
from the prior record-stack export. The lower qMLP vocab ladder points cannot
run until their matching CaseOps tokenizer model, train shards, validation
shard, and validation-byte sidecar exist.

## Current Assumptions

- Use the 04-23 record-stack CaseOps transform and prep path.
- `sp8192` and `sp16384` are already built and should be reused.
- Missing vocab sets are independent and should be exported in parallel when
  scheduler resources allow.
- The user explicitly allowed up to 32 CPUs for this export goal, so 32-CPU
  Slurm allocations are in scope even though the generic agent guide asks for
  approval above 16 CPUs.

## Remote Export Matrix

Checked on `submit-a.ib.coehpc` at `2026-06-22T23:34:02-07:00`.

| Vocab | Export Root | Tokenizer | Train Shards | Val Shards | Val Byte Sidecars | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `1024` | `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp1024` | missing | 0 | 0 | 0 | building |
| `2048` | `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp2048` | missing | 0 | 0 | 0 | building |
| `4096` | `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp4096` | missing | 0 | 0 | 0 | building |
| `8192` | `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched` | present | 80 | 1 | 1 | ready |
| `16384` | `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384` | present | 80 | 1 | 1 | ready |

## Build Strategy

Use the proven remote script:

```text
/nfs/hpc/share/peterj29/pg/src/pg/goal/15-caseops-vocab.sbatch
```

Submit one independent Slurm job per missing vocab. This is faster than a
serial vocab ladder because the exports do not share mutable inputs and each
writes a distinct `caseops-sp<VOCAB>` root.

Initial resource choice:

```text
partition=share
time=03:00:00
cpus-per-task=32
mem=96G
TOKENIZER_TRAIN_DOCS=500000
MAX_TRAIN_SHARDS=80
VAL_DOCS=10000
MAX_DOCS=0
```

This keeps memory below the generic 128 GB guide ceiling while using the
user-approved 32 CPU limit.

## Submitted Jobs

Run root:

```text
/nfs/hpc/share/peterj29/pg/runs/goal2-phase1-caseops-data
```

| Vocab | Job ID | Job Name | CPUs | Memory | Node/State at Submit Follow-up |
| --- | ---: | --- | ---: | ---: | --- |
| `1024` | `20485659` | `pg-g2-cops1024` | 32 | 96G | running on `cn-r-4` |
| `2048` | `20485660` | `pg-g2-cops2048` | 32 | 96G | running on `cn-r-5` |
| `4096` | `20485661` | `pg-g2-cops4096` | 32 | 96G | running on `cn-d11` |

An attempted smaller replacement for `sp4096`, job `20485667`, was cancelled
after `00:00:14` once the original 32-CPU job had started. Read-only inspection
showed it left only its own run manifest/log files and no files under
`caseops-sp4096`.

## Completion Requirements

This phase is complete when every target vocab has:

- tokenizer `.model`;
- tokenizer `.vocab` when produced by the trainer;
- 80 `fineweb_train_*.bin` shards;
- at least one `fineweb_val_*.bin` shard;
- at least one `fineweb_val_bytes_*.bin` sidecar;
- terminal Slurm state `COMPLETED` with exit code `0:0`, or a captured blocker.

## Result

Status: in progress

Evidence:

- `sp8192` and `sp16384` are verified ready.
- `sp1024`, `sp2048`, and `sp4096` were missing and have active Slurm export
  jobs.

Artifacts:

- Remote run root:
  `/nfs/hpc/share/peterj29/pg/runs/goal2-phase1-caseops-data`

Decision:

- Monitor jobs `20485659`, `20485660`, and `20485661`.
- After terminal states, verify artifact counts directly under each export root.
