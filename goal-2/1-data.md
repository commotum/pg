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
32768
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

Checked on `submit-a.ib.coehpc` at `2026-06-23T00:18-00:23 PDT`.

| Vocab | Export Root | Tokenizer | Train Shards | Val Shards | Val Byte Sidecars | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `1024` | `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp1024` | present | 80 | 2 | 2 | ready |
| `2048` | `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp2048` | present | 80 | 2 | 2 | ready |
| `4096` | `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp4096` | present | 80 | 2 | 2 | ready |
| `8192` | `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched` | present | 80 | 1 | 1 | ready |
| `16384` | `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384` | present | 80 | 1 | 1 | ready |
| `32768` | `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp32768` | present | 80 | 1 | 1 | ready |

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

| Vocab | Job ID | Job Name | CPUs | Memory | Node | Final State | Elapsed |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: |
| `1024` | `20485659` | `pg-g2-cops1024` | 32 | 96G | `cn-r-4` | `COMPLETED`, exit `0:0` | `00:42:30` |
| `2048` | `20485660` | `pg-g2-cops2048` | 32 | 96G | `cn-r-5` | `COMPLETED`, exit `0:0` | `00:37:23` |
| `4096` | `20485661` | `pg-g2-cops4096` | 32 | 96G | `cn-d11` | `COMPLETED`, exit `0:0` | `01:14:16` |
| `32768` | `20486174` | `pg-g2-cops32768` | 32 | 96G | `cn-d11` | `COMPLETED`, exit `0:0` | `01:33:58` |

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

Status: reopened for `sp32768`

Evidence:

- `sp8192` and `sp16384` are verified ready.
- `sp2048` export job `20485660` completed with exit code `0:0` after
  `00:37:23`; direct counts found 80 train shards, two validation shards, two
  validation-byte sidecars, one tokenizer model, one tokenizer vocab, and one
  tokenizer manifest.
- `sp1024` export job `20485659` completed with exit code `0:0` after
  `00:42:30`; direct counts found 80 train shards, two validation shards, two
  validation-byte sidecars, one tokenizer model, one tokenizer vocab, and one
  tokenizer manifest. It has been released to Phase 3 smokes.
- `sp4096` export job `20485661` completed with exit code `0:0` after
  `01:14:16`; direct counts found 80 train shards, two validation shards, two
  validation-byte sidecars, one tokenizer model, one tokenizer vocab, and one
  tokenizer manifest. It has been released to Phase 3 smokes.
- Final verified matrix:
  - `sp1024`: `tok=yes vocab=yes manifest=yes train=80 val=2 val_bytes=2`
  - `sp2048`: `tok=yes vocab=yes manifest=yes train=80 val=2 val_bytes=2`
  - `sp4096`: `tok=yes vocab=yes manifest=yes train=80 val=2 val_bytes=2`
  - `sp8192`: `tok=yes train=80 val=1 val_bytes=1`
  - `sp16384`: `tok=yes vocab=yes manifest=yes train=80 val=1 val_bytes=1`
- User direction on 2026-06-23 added `sp32768` to the target matrix for both
  dense and qMLP, three benchmark seeds each. Direct remote inspection found no
  existing `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp32768` or patched
  equivalent, so a new CaseOps export is required before smokes can run.
- Submitted `sp32768` export job `20486174` with 32 CPUs, 96 GB memory, and the
  standard 80-shard CaseOps export settings.
- `sp32768` export job `20486174` completed on `cn-d11` with exit `0:0` after
  `01:33:58`.
- The target export root
  `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp32768` now has tokenizer
  model and vocab files. The final read-only artifact count found
  `tok=yes vocab=yes train=80 val=1 val_bytes=1`, so the `sp32768` CaseOps
  export is ready for smoke and benchmark use.

Artifacts:

- Remote run root:
  `/nfs/hpc/share/peterj29/pg/runs/goal2-phase1-caseops-data`
- New export roots:
  - `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp1024`
  - `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp2048`
  - `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp4096`
  - `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp32768`

Decision:

- Phase 1 data dependencies are complete for all six CaseOps vocabs, including
  `sp32768`.
