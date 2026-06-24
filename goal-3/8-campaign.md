# Phase 8: Campaign

## Overview

Run one autonomous 8xH100 H100/FA3 campaign allocation. The campaign must use
the known record stack as a trust check before spending the allocation on qMLP.

Primary script:

```text
goal-3/h100-campaign-runner.sbatch
```

The H100 campaign has been submitted as Slurm job `20487886` after explicit
user approval.

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
6. Run short distributed candidate smokes:
   - `dense_sp8192_smoke`, seed `42`;
   - `qmlp_sp8192_smoke`, seed `42`;
   - `qmlp_sp16384_smoke`, seed `42`;
   - `qmlp_sp16384_ttt_smoke`, seed `42`.
7. Write `smoke-gate.json` and stop if any smoke fails. If one smoke fails,
   later smoke candidates are marked `not_attempted` instead of running.
8. Run full dense/base `sp8192`, seed `42`.
9. Write `baseline-parity.json`, including when the baseline candidate exits
   nonzero.
10. Stop only if the baseline fails the hard validity gate. If it misses the
   stricter parity target but remains credible, record the caveat and continue.
11. Run full qMLP `sp16384`, seed `42`.
12. Run full qMLP `sp16384`, seed `0`.
13. Run full qMLP `sp16384`, seed `1234`.
14. Write `final-status.json` with all run summaries and qMLP mean/std.

## Baseline Parity Gate

Defaults:

```text
GOAL3_BASELINE_CANDIDATE=dense_sp8192
GOAL3_BASELINE_SEED=42
GOAL3_SMOKE_CANDIDATES="dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384_smoke qmlp_sp16384_ttt_smoke"
GOAL3_SMOKE_SEED=42
GOAL3_SMOKE_TIMEOUT=20m
strict parity target:
  GOAL3_BASELINE_PARITY_MAX_BPB=1.065
  GOAL3_BASELINE_PARITY_MIN_STEPS=4500

hard stop gate:
  GOAL3_BASELINE_HARD_MAX_BPB=1.075
  GOAL3_BASELINE_HARD_MIN_STEPS=4000

GOAL3_FULL_TIMEOUT=120m
```

The qMLP seeds must not run if the dense baseline path is clearly invalid. A
borderline baseline should not waste the allocation; it should continue with a
recorded comparison caveat unless the hard gate fails.

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
- `runtime-storage.txt`;
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
- `smoke-gate.json`;
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
If the campaign exits early, `final-status.json` must still collect whatever
partial evidence exists from env smoke, smoke gate, baseline parity, candidate
summaries, candidate statuses, artifact manifests, and the source snapshot
manifest.

## Completion Requirements

- H100 job reaches a terminal Slurm state.
- Runtime setup result is recorded.
- Environment smoke result is recorded.
- Short dense/qMLP candidate smoke gate result is recorded.
- Dense baseline parity result is recorded.
- If the baseline hard validity gate passes, all three qMLP seeds are attempted
  in order.
- Logs, summaries, artifacts, and hashes remain in shared storage.
- `goal-3/jobs.csv`, `goal-3/status.md`, and `goal-3/findings-summary.md` are
  updated from the actual run results.

## Current State

Queued:

- Slurm job `20487886`, script `goal-3/h100-campaign-runner.sbatch`;
- latest checked state `PD`, reason `(Priority)`;
- requested resources: `dgxh`, `h100&vram80g`, `gpu:8`, `nodes=1`,
  `ntasks=1`, `cpus-per-task=64`, `mem=500G`, `time=06:00:00`;
- run directory:
  `/nfs/hpc/share/peterj29/pg/goal-3-runs/goal3-h100-campaign-20487886`.

Complete:

- user approval received;
- Slurm submission complete;
- local static checks passed after campaign edits;
- `goal-3/` synced to the remote HPC checkout;
- remote submit-node static checks passed;
- exact six-hour `srun --test-only` refresh immediately before submission
  passed as dry-run job `20487885`, with predicted start
  `2026-06-28T08:29:30` on `dgxh-3`;
- eight-hour comparison dry-run was rejected by `MaxGRESRunMinsPerUser`, so
  six hours is the visible 8xH100 QOS ceiling.
