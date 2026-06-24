# Phase 6: Runner

## Overview

Create the unattended H100 runner and bounded repair-agent entry point. These
scripts are ready for review and static validation, but they have not been
submitted.

## Scripted Artifacts

Primary campaign runner:

```text
goal-3/h100-campaign-runner.sbatch
```

Legacy/component record runner:

```text
goal-3/h100-record-runner.sbatch
```

Bounded repair agent:

```text
goal-3/h100-repair-agent.sbatch
```

Shared candidate runner:

```text
goal-3/scripts/run_candidate.sh
```

Inside Slurm allocations, `run_candidate.sh` launches the candidate workload as:

```text
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This keeps the training/package step under Slurm accounting while still letting
`torchrun` own the single-node eight-process distributed launch. The behavior
can be disabled only with `GOAL3_USE_SRUN_LAUNCHER=0`, which should be treated
as a reviewed exception.

Parser:

```text
goal-3/scripts/parse_train_log.py
```

Static Goal 3 guardrail audit:

```text
goal-3/scripts/static_goal3_audit.py
```

This script does not import the record stack. It checks text-level invariants
for qMLP wiring, candidate mapping, H100 80GB constraints, final-status traps,
scratch staging, and bounded default candidate order/timeouts.
It intentionally avoids newer Python-only syntax so it can run under the older
submit-node `python3` used for static checks.

Environment smoke:

```text
goal-3/scripts/env_smoke.py
```

## Default H100 Campaign Request

The campaign runner currently requests:

```text
partition: dgxh
constraint: h100&vram80g
gres: gpu:8
nodes: 1
ntasks: 1
cpus-per-task: 64
memory: 500G
walltime: 06:00:00
```

The 80GB constraint is intentional. Phase 0 live checks showed
`--constraint=h100` alone could target `dgxh-1` with `gpu:h100-40g:16`, which is
not the intended competition-class node.

## Default Campaign Order

The campaign runner default is:

```text
runtime_setup_and_env_smoke
dense_sp8192_smoke/qmlp_sp8192_smoke/qmlp_sp16384_smoke
dense_sp8192 seed 42 baseline parity
qmlp_sp16384 seed 42
qmlp_sp16384 seed 0
qmlp_sp16384 seed 1234
```

Default campaign gates:

```text
strict parity target:
  GOAL3_BASELINE_PARITY_MAX_BPB=1.065
  GOAL3_BASELINE_PARITY_MIN_STEPS=4500

hard stop gate:
  GOAL3_BASELINE_HARD_MAX_BPB=1.075
  GOAL3_BASELINE_HARD_MIN_STEPS=4000

GOAL3_FULL_TIMEOUT=120m
GOAL3_SMOKE_TIMEOUT=20m
```

The `GOAL3_FULL_TIMEOUT` is much larger than the 10-minute training budget
because the full candidate includes quantization, TTT/eval, compression,
parsing, and stage-out. It is a stuck-candidate backstop, not an expected
runtime target. The record script still uses
`MAX_WALLCLOCK_SECONDS=600` for the training budget.

Reasoning:

- runtime setup/env smoke catches missing FA3, CUDA, `lrzip`, and tokenizer
  issues inside the actual H100 allocation;
- short dense/qMLP/qMLP+TTT candidate smokes catch distributed launch, qMLP
  wiring, package accounting, tokenizer/vocab issues, and TTT LoRA hook issues
  before the full baseline and qMLP runs;
- full `dense_sp8192` seed 42 checks that the OSU setup reproduces the known
  record path closely enough to trust qMLP results;
- qMLP `sp16384` seeds 42, 0, and 1234 are the actual record-attempt campaign.

The campaign can be overridden only as part of a reviewed H100 request:

```bash
GOAL3_QMLP_SEEDS="42 0 1234" sbatch goal-3/h100-campaign-runner.sbatch
```

The older `goal-3/h100-record-runner.sbatch` remains a one-hour component
runner with smoke-oriented defaults. It is not the recommended approval target.
The separate `goal-3/h100-short-smoke.sbatch` remains available for fallback
diagnosis only after explicit review.

## Outputs

Each run writes to:

```text
/nfs/hpc/share/peterj29/pg/goal-3-runs/<job-name>-<job-id>/
```

Expected files:

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
- `smoke-gate.json`;
- `baseline-parity.json`;
- `candidates/<candidate>/seed_<seed>/stdout.log`;
- `candidates/<candidate>/seed_<seed>/stderr.log`;
- `candidates/<candidate>/seed_<seed>/env.txt`;
- `candidates/<candidate>/seed_<seed>/summary.json`;
- `candidates/<candidate>/seed_<seed>/status.json`;
- `candidates/<candidate>/seed_<seed>/artifacts.json`;
- `final-status.json`.

Each candidate `env.txt` records the effective launcher as either `direct` or
the quoted `srun ...` command.
Each candidate `status.json` records `exit_code`, `timeout`, and `timed_out` so
a bounded timeout is visible without reading the whole stderr log.
Each candidate `artifacts.json` records byte size and sha256 for each non-log
file in the candidate directory.
The H100 env, short-smoke, record-runner, campaign-runner, and optional
repair-agent scripts install an exit trap that writes `final-status.json` on
early failure or signal if the normal summary path has not already written one.
The short-smoke and record-runner normal summaries also include candidate order,
seed, timeout settings, and the per-candidate `status.json` payloads.
The campaign runner writes a normal `final-status.json` with
`smoke_gate`, `baseline_parity`, all candidate summaries/statuses/artifact
manifests, and qMLP post-TTT BPB mean/std when all qMLP seeds complete.
On early failure, the campaign runner also writes a campaign-aware
`final-status.json` that includes whatever partial env-smoke, smoke-gate,
baseline-parity, candidate, artifact, and source-snapshot evidence exists.

## Repair Agent

`goal-3/h100-repair-agent.sbatch` exits immediately unless:

```bash
GOAL3_ENABLE_REPAIR_AGENT=1
```

If enabled, it runs:

```text
codex exec
```

with:

- a timeout;
- the Goal 3 prompt plus repair-agent bounds;
- logs in the current run directory;
- no permission to submit new H100 jobs;
- no broad sweeps;
- no destructive cleanup;
- at most one bounded smoke after a patch.

This is not the default execution path.

## Verification

Static checks completed:

```bash
bash -n goal-3/h100-campaign-runner.sbatch
bash -n goal-3/h100-record-runner.sbatch
bash -n goal-3/h100-repair-agent.sbatch
bash -n goal-3/scripts/common.sh
bash -n goal-3/scripts/run_candidate.sh
python3 -m py_compile goal-3/scripts/parse_train_log.py
python3 -m py_compile goal-3/scripts/env_smoke.py
python3 -m py_compile goal-3/scripts/static_goal3_audit.py
python3 goal-3/scripts/static_goal3_audit.py
```

Runtime checks pending:

- no H100 script has been submitted;
- no final runner output exists yet.

Runtime checks completed:

- `srun --test-only` was refreshed on 2026-06-23 at 16:39 Pacific for the old
  15-minute env smoke and one-hour record runner. Those dry-runs are now stale
  for approval because the target changed to the padded campaign runner.
- `srun --test-only` for the exact six-hour campaign request must be refreshed
  after the padding change.
- Local `bash -n` checks were rerun on 2026-06-23 at 16:30 Pacific after the
  final-status trap and summary updates.
- Local `static_goal3_audit.py` passed on 2026-06-23 at 16:35 Pacific.
- Remote submit-node static checks, including `static_goal3_audit.py`, passed
  on 2026-06-23 at 16:38 Pacific after syncing through the OSU gateway.
- Remote submit-node static checks passed again on 2026-06-23 after the campaign
  runner updates.
- Local static checks passed on 2026-06-23 at 18:23 Pacific after adding the
  `qmlp_sp16384_ttt_smoke` default to the standalone short-smoke fallback and
  extending the static audit for that invariant.
- Remote submit-node static checks passed on 2026-06-23 at 18:23 Pacific after
  syncing the same qMLP+TTT smoke/default update through the OSU gateway.

## Scratch Staging

The short-smoke and record-runner scripts now call
`goal3_prepare_local_workspace`, which stages:

- `goal-3/stage/primary-qmlp`;
- CaseOps `sp8192` data and tokenizer;
- CaseOps `sp16384` data and tokenizer.

The staged paths live under:

```text
/scratch/$USER/$SLURM_JOB_ID/goal3
```

The runner records the staged mapping in `scratch-stage.txt` and uses a guarded
cleanup trap that refuses to remove paths outside the expected job scratch
directory.

## Completion Requirements

- Campaign runner exists: complete.
- Runner has explicit resource request: complete.
- Runner has short dense/qMLP/qMLP+TTT smoke gate, baseline hard validity gate,
  strict parity note, and bounded qMLP seed order: complete.
- Runner writes machine-readable final status: complete.
- Repair agent exists and is disabled by default: complete.
- H100 campaign dry-run is current: complete.
- Static Goal 3 audit passes locally: complete.
- Remote submit-node static checks pass after the latest sync: complete.
- User approval for exact H100 request: pending.

## Next Phase

Phase 7 is the human approval gate. Before submitting H100 work, refresh live
Slurm state, run `srun --test-only` with the exact six-hour campaign request,
show the script path and candidate order to the user, and wait for explicit
approval.
