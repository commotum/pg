# Goal 3 Compliance Note

## Scope

This note covers the staged Goal 3 qMLP implementation in:

```text
goal-3/stage/primary-qmlp/train_gpt.py
```

It compares against the selected base:

```text
parameter-golf/records/track_10min_16mb/
2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/train_gpt.py
```

## qMLP Change

The staged code adds an opt-in qMLP MLP parameterization:

```text
QUAT_MLP=0  dense/base path
QUAT_MLP=1  qMLP path
```

When qMLP is enabled, MLP up/down banks are stored as four compact quaternion
component matrices and materialized into Hamilton dense matrices before the
existing fused LeakyReLU-square MLP compute path.

## Compliance-Sensitive Surfaces Preserved

Data access is unchanged:

- the same `DATA_PATH` mechanism is used;
- the same CaseOps validation byte sidecar mechanism is used;
- dense and qMLP runs at the same vocab use the same tokenizer, train shards,
  validation shard, and validation byte sidecar;
- qMLP does not add external data, cached labels, validation-derived training
  data, or cross-document context.

Timing/accounting is unchanged in structure:

- the same `MAX_WALLCLOCK_SECONDS` training cap is used;
- the same GPTQ reserve mechanism is used;
- the same serializer reports total submission bytes;
- the same post-quant and TTT eval paths produce BPB.

Evaluation semantics are unchanged in structure:

- score-first TTT remains the selected record-stack TTT path;
- document-boundary handling remains in the base validation and TTT code;
- CaseOps original-byte BPB accounting remains through `fineweb_val_bytes_*.bin`.

Packaging semantics are unchanged in structure:

- qMLP tensors are routed through the existing GPTQ metadata format;
- qMLP component tensors are included in unbank/rebank;
- qMLP component tensor groups are included in per-group compression;
- total submission bytes are still parsed from the training script output.

## Required Runtime Proof

This note does not claim final compliance. Runtime proof still requires:

- the staged qMLP candidate serializes successfully;
- total submission bytes are below `16,000,000`;
- BPB is computed from the CaseOps byte sidecar;
- the job logs record exact host, GPUs, seed, data paths, command, Git state,
  and artifact paths;
- the run reaches a terminal Slurm state without unplanned data access or
  unplanned H100 debugging.

If any of those checks fail or are ambiguous, the candidate is not compliant
until the issue is fixed and rerun.
