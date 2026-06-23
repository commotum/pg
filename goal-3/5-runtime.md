# Phase 5: Runtime

## Overview

Catch ordinary runtime bugs before a full record attempt. This phase is not
complete yet because the selected record stack depends on CUDA, FA3/Triton,
`lrzip`, and the CaseOps data environment.

## Current Runtime Position

Completed locally:

- staged `train_gpt.py` syntax compile;
- helper parser syntax compile;
- sbatch/helper `bash -n` checks;
- reusable environment smoke script created.

Pending on HPC:

- build or verify the shared Python environment with `goal-3/prepare-env.sbatch`;
- build or verify user-local `lrzip` with `goal-3/prepare-tools.sbatch`;
- run `goal-3/h100-env-smoke.sbatch` only after H100 approval;
- run `goal-3/h100-short-smoke.sbatch` only after H100 approval.

Completed on HPC submit node:

- synced local Goal 3 files to `/nfs/hpc/share/peterj29/pg/src/pg/goal-3`;
- remote `bash -n` checks passed for Goal 3 shell scripts;
- remote Python `compile(...)` syntax checks passed for the Goal 3 Python
  scripts and staged `train_gpt.py`.

## Why No Local Runtime Smoke

The full selected record stack imports CUDA/H100-specific dependencies at module
load time, including `flash_attn_interface` and Triton. A local Mac or submit
node smoke would either fail for irrelevant reasons or encourage mutating a
submit-node Python environment. That would not prove the H100 path.

The next useful non-H100 runtime preparation is the CPU Slurm environment build,
not model execution.

## Verification

Static verification currently available:

```bash
python3 -m py_compile goal-3/stage/primary-qmlp/train_gpt.py
python3 -m py_compile goal-3/scripts/parse_train_log.py
python3 -m py_compile goal-3/scripts/env_smoke.py
bash -n goal-3/scripts/common.sh
bash -n goal-3/scripts/run_candidate.sh
bash -n goal-3/prepare-env.sbatch
bash -n goal-3/h100-env-smoke.sbatch
bash -n goal-3/h100-short-smoke.sbatch
bash -n goal-3/h100-record-runner.sbatch
bash -n goal-3/h100-repair-agent.sbatch
```

Runtime verification still needed:

- environment preparation reaches terminal Slurm state;
- tools preparation reaches terminal Slurm state or another `lrzip` path is
  documented;
- H100 env smoke confirms 8 CUDA devices, FA3 import, `lrzip`, and tokenizer
  vocab sizes;
- H100 short smoke confirms dense `sp8192`, qMLP `sp8192`, and qMLP `sp16384`
  can start and serialize/package.

## Completion Requirements

- CPU environment-prep job submitted and terminal: pending.
- CPU tools-prep job submitted and terminal: pending.
- H100 env smoke approved, submitted, and terminal: pending.
- H100 short qMLP smoke approved, submitted, and terminal: pending.
- Runtime logs and parser summaries recorded in `goal-3/jobs.csv`: pending.

## Next Phase

Do not request H100 yet. First prepare the shared Python environment and
user-local `lrzip` through CPU Slurm jobs.
