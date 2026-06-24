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

- run the campaign-internal H100 environment smoke only after approval of
  `goal-3/h100-campaign-runner.sbatch`;
- keep `goal-3/h100-env-smoke.sbatch` and `goal-3/h100-short-smoke.sbatch` as
  fallback diagnostics, not the default next H100 request.

Completed on HPC CPU compute nodes:

- `goal-3/prepare-env.sbatch` completed as Slurm job `20487397` on `cn-r-6`;
  it built `/nfs/hpc/share/peterj29/pg/envs/goal3-cu128` with Python 3.12,
  `torch==2.9.1+cu128`, Triton, `sentencepiece`, `brotli`, and
  `flash_attn_3`;
- `goal-3/prepare-tools.sbatch` completed as Slurm job `20487617` on `cn-r-5`;
  it built `/nfs/hpc/share/peterj29/pg/tools/lrzip/bin/lrzip` using
  user-local LZO and LZ4 libraries.

Important caveat: running the compute-built `lrzip` binary directly on the
submit node fails because the submit node exposes an older glibc. That does not
invalidate the compute-node build, but it means the next meaningful check is
the H100 env smoke inside the H100 allocation.

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

The useful non-H100 runtime preparation has already been done: the shared Python
environment and user-local `lrzip` tool were built on CPU Slurm compute nodes.

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
bash -n goal-3/h100-campaign-runner.sbatch
bash -n goal-3/h100-repair-agent.sbatch
```

Runtime verification still needed:

- campaign-internal H100 env smoke confirms 8 CUDA devices, FA3 import,
  `lrzip`, and tokenizer vocab sizes;
- full dense `sp8192` baseline parity confirms the OSU H100 setup is credible
  before spending time on qMLP;
- qMLP `sp16384` seeds confirm runtime, package accounting, and record-level
  metrics.

## Completion Requirements

- CPU environment-prep job submitted and terminal: complete, job `20487397`.
- CPU tools-prep job submitted and terminal: complete, job `20487617`.
- H100 campaign approved, submitted, and terminal: pending.
- Campaign baseline parity complete: pending.
- Campaign qMLP seed runs complete: pending.
- Runtime logs and parser summaries recorded in `goal-3/jobs.csv`: partial.

## Next Phase

The next runtime step is approval of the autonomous H100 campaign runner. Its
first runtime gate is the env smoke, followed by dense baseline parity and then
the qMLP seeds.
