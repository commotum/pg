# Phase 8: Campaign

## Overview

Run one autonomous 8xH100 H100/FA3 campaign allocation. The campaign must use
the known record stack as a trust check before spending the allocation on qMLP.

Primary script:

```text
goal-3/h100-campaign-runner.sbatch
```

No H100/H200 job has been submitted yet.

## Required Order

1. Activate `/nfs/hpc/share/peterj29/pg/envs/goal3-cu128`.
2. Validate/build runtime requirements:
   - record imports before setup;
   - install FA3 from the documented wheel source if `flash_attn_interface` is
     missing and `GOAL3_ALLOW_RUNTIME_FA3_INSTALL=1`;
   - record imports after setup;
   - fail if required modules or `lrzip` remain missing.
3. Record context:
   - host;
   - Slurm environment;
   - Git SHA and dirty diff;
   - module list;
   - GPU inventory;
   - Python dependency context.
4. Run `goal-3/scripts/env_smoke.py` under `srun`.
5. Stage Goal 3 source and CaseOps `sp8192`/`sp16384` data to node-local
   scratch.
6. Run full dense/base `sp8192`, seed `42`.
7. Write `baseline-parity.json`.
8. Stop if baseline parity fails.
9. Run full qMLP `sp16384`, seed `42`.
10. Run full qMLP `sp16384`, seed `0`.
11. Run full qMLP `sp16384`, seed `1234`.
12. Write `final-status.json` with all run summaries and qMLP mean/std.

## Baseline Parity Gate

Defaults:

```text
GOAL3_BASELINE_CANDIDATE=dense_sp8192
GOAL3_BASELINE_SEED=42
GOAL3_BASELINE_PARITY_MAX_BPB=1.065
GOAL3_BASELINE_PARITY_MIN_STEPS=4500
GOAL3_FULL_TIMEOUT=30m
```

The qMLP seeds must not run if the dense baseline path is not credible. A bad
baseline means the OSU H100 setup, package path, or runtime stack is not trusted
for record claims.

## qMLP Candidate

Defaults:

```text
GOAL3_QMLP_CANDIDATE=qmlp_sp16384
GOAL3_QMLP_SEEDS="42 0 1234"
```

The campaign is not a sweep. Additional candidates require a revised approval
packet.

## Required Outputs

Preserve these under:

```text
/nfs/hpc/share/peterj29/pg/goal-3-runs/goal3-h100-campaign-$SLURM_JOB_ID
```

Required files:

- `context.txt`;
- `gpu.txt`;
- `python.txt`;
- `runtime-setup/imports-before.json`;
- `runtime-setup/imports-after.json`;
- `runtime-setup/fa3-install.txt`, if runtime FA3 install ran;
- `env-smoke.json`;
- `git-status.txt`;
- `git-diff.stat`;
- `git-diff.patch`;
- `source-snapshot/goal-3/`;
- `source-snapshot.sha256`;
- `scratch-stage.txt`;
- `run-order.txt`;
- `progress.txt`;
- `baseline-parity.json`;
- per-candidate `stdout.log`;
- per-candidate `stderr.log`;
- per-candidate `env.txt`;
- per-candidate `summary.json`;
- per-candidate `status.json`;
- per-candidate `artifacts.json`;
- `final-status.json`.

The `artifacts.json` manifests must include byte sizes and sha256 hashes for
the full model/submission outputs preserved in the candidate directories.

## Completion Requirements

- H100 job reaches a terminal Slurm state.
- Runtime setup result is recorded.
- Environment smoke result is recorded.
- Dense baseline parity result is recorded.
- If baseline parity passes, all three qMLP seeds are attempted in order.
- Logs, summaries, artifacts, and hashes remain in shared storage.
- `goal-3/jobs.csv`, `goal-3/status.md`, and `goal-3/findings-summary.md` are
  updated from the actual run results.

## Current State

Pending:

- user approval;
- Slurm submission.

Complete:

- local static checks passed after campaign edits;
- `goal-3/` synced to the remote HPC checkout;
- remote submit-node static checks passed;
- exact three-hour `srun --test-only` predicts `dgxh-3` at
  `2026-06-27T20:29:30`.
