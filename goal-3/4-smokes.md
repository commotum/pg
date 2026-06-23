# Phase 4: Smokes

## Overview

Prepare package/runtime smoke entry points that can verify the staged qMLP
record stack before a full record attempt. These scripts are created and
statically checked, but they have not been submitted.

The full 2026-04-27 record stack depends on CUDA, FA3, Triton, `lrzip`,
`sentencepiece`, and the CaseOps datasets. Runtime verification must happen
inside Slurm compute allocations.

## Scripted Artifacts

Environment preparation:

```text
goal-3/prepare-env.sbatch
```

H100 environment smoke:

```text
goal-3/h100-env-smoke.sbatch
```

Short H100 candidate smoke:

```text
goal-3/h100-short-smoke.sbatch
```

Shared helpers:

```text
goal-3/scripts/common.sh
goal-3/scripts/run_candidate.sh
goal-3/scripts/parse_train_log.py
goal-3/scripts/env_smoke.py
```

## Candidate Smokes

The short smoke script defaults to:

```text
dense_sp8192_smoke
qmlp_sp8192_smoke
qmlp_sp16384_smoke
```

Each smoke uses a bounded tiny run:

```text
ITERATIONS=20
MAX_WALLCLOCK_SECONDS=120
WARMUP_STEPS=1
VAL_DOC_FRACTION=0.01
GPTQ_CALIBRATION_BATCHES=1
TTT_ENABLED=0
```

The smoke still exercises:

- tokenizer load and vocab-size check through `ValidationData`;
- train shard discovery;
- distributed `torchrun`;
- qMLP forward/backward for qMLP candidates;
- GPTQ/serialization/package accounting unless the run fails earlier;
- parser extraction into `summary.json` and `status.json`.

The H100 smoke scripts also run `goal-3/scripts/env_smoke.py` before candidate
execution to check:

- 8 visible CUDA devices;
- `torch`, `triton`, `sentencepiece`, `brotli`, and `flash_attn_interface`;
- `lrzip` on `PATH`;
- `sp8192` and `sp16384` tokenizer vocab sizes.

## Verification

Static checks completed:

```bash
bash -n goal-3/scripts/common.sh
bash -n goal-3/scripts/run_candidate.sh
python3 -m py_compile goal-3/scripts/parse_train_log.py
python3 -m py_compile goal-3/scripts/env_smoke.py
bash -n goal-3/prepare-env.sbatch
bash -n goal-3/h100-env-smoke.sbatch
bash -n goal-3/h100-short-smoke.sbatch
```

Runtime checks pending:

- `prepare-env.sbatch` has not been submitted;
- `h100-env-smoke.sbatch` has not been submitted;
- `h100-short-smoke.sbatch` has not been submitted;
- no package bytes have been measured from the staged qMLP file yet.

## Completion Requirements

- Smoke scripts exist: complete.
- Static checks pass: complete.
- Shared parser exists: complete.
- Environment build succeeds on HPC: pending.
- H100 env smoke passes: pending user-approved H100 job.
- Candidate smoke package bytes are known: pending user-approved H100 job.

## Next Phase

Phase 5 should run the cheapest approved non-H100 work first: sync the repo to
HPC and submit the CPU environment-prep job if the user wants to proceed. H100
smokes remain gated behind explicit review and approval.
