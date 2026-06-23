# Phase 2: Data

## Overview

Verify that the CaseOps tokenizers, train shards, validation shards, and
original-byte sidecars needed for the first H100 qMLP campaign already exist.
This phase is read-only and does not run data export.

## Required Vocabularies

Goal 3 requires:

- `sp8192` CaseOps for the exact base/dense smoke and qMLP same-vocab control;
- `sp16384` CaseOps for the budget-reinvested qMLP candidate.

Dense/base and qMLP runs at the same vocab must use the exact same tokenizer,
train shards, validation tokens, and validation-byte sidecars.

## Verified Manifests

Machine-readable manifest:

```text
goal-3/data-manifest.csv
```

### CaseOps `sp8192`

Label:

```text
caseops-sp8192-patched
```

Tokenizer:

```text
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

Tokenizer facts:

```text
bytes: 366,510
sha256: 97754754561f20def5edd99765ca0668f880bd94e67316d54aa10ab25f45cc4d
```

The remote tokenizer hash matches both local shipped record tokenizers:

```text
2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
2026-04-29_SmearGateBOSFix_3Seed_1.06141/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

Data path:

```text
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
```

Data facts:

```text
train shards: 80
validation token shard: fineweb_val_000000.bin, 19,326,028 bytes
validation byte sidecar: fineweb_val_bytes_000000.bin, 19,326,028 bytes
dataset directory size: 1.6G
creation job: unknown from current local docs
```

### CaseOps `sp16384`

Label:

```text
caseops-sp16384
```

Tokenizer:

```text
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/datasets/tokenizers/fineweb_16384_bpe_lossless_caps_caseops_v1_reserved.model
```

Tokenizer facts:

```text
bytes: 506,698
sha256: 906026c0d84eaa96285fdacf250c529555bb135e9fe2e6b16793e595130eab19
```

Additional tokenizer files:

```text
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/datasets/tokenizers/fineweb_16384_bpe_lossless_caps_caseops_v1_reserved.vocab
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/datasets/tokenizers/fineweb_16384_bpe_lossless_caps_caseops_v1_reserved.manifest.json
```

Data path:

```text
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/datasets/datasets/fineweb10B_sp16384_lossless_caps_caseops_v1_reserved
```

Data facts:

```text
train shards: 80
validation token shard: fineweb_val_000000.bin, 18,070,246 bytes
validation byte sidecar: fineweb_val_bytes_000000.bin, 18,070,246 bytes
dataset directory size: 1.6G
creation job: unknown from current local docs
```

## Verification Commands

Read-only checks performed through the OSU submit node:

```text
find /nfs/hpc/share/peterj29/pg/data-exports -maxdepth 2 -type d
find exact caseops-sp8192-patched path -type f
find exact caseops-sp16384 path -type f
stat tokenizer and validation files
du -sh dataset directories
sha256sum tokenizer models
```

Local checks:

```text
stat local shipped sp8192 tokenizer models
shasum -a 256 local shipped sp8192 tokenizer models
```

The submit-node system Python does not have `sentencepiece`, so direct
`SentencePieceProcessor(...).vocab_size()` verification is deferred to the
environment smoke rather than mutating the submit-node environment.

## Findings

- Both required CaseOps exports already exist on HPC shared storage.
- No new CPU Slurm export is needed for the first Goal 3 H100 campaign.
- The `sp8192` remote tokenizer matches the primary and fallback record-shipped
  tokenizer by SHA-256.
- The `sp16384` export has tokenizer model, vocab, and tokenizer manifest files.
- Both exports include validation byte sidecars needed for original-byte BPB.
- Both exports use a nested `datasets/datasets/...` layout; H100 scripts must
  pass the inner dataset directory as `DATA_PATH`.

## Completion Requirements

- `sp8192` manifest exists: complete in `goal-3/data-manifest.csv`.
- `sp16384` manifest exists: complete in `goal-3/data-manifest.csv`.
- Dense/qMLP same-vocab pairing policy is explicit: complete.
- No data export ran on a submit node: complete.
- Missing data is either produced or documented as a blocker: no missing data
  found for first campaign.

## Next Phase

Phase 3: port qMLP into the selected 2026-04-27 record stack, using the banked
qMLP implementation pattern from the 2026-04-23 record.
