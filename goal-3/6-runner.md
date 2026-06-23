# Phase 6: Runner

## Overview

Create the unattended H100 runner and bounded repair-agent entry point. These
scripts are ready for review and static validation, but they have not been
submitted.

## Scripted Artifacts

Final runner:

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

Parser:

```text
goal-3/scripts/parse_train_log.py
```

Environment smoke:

```text
goal-3/scripts/env_smoke.py
```

## Default H100 Request

The record runner currently requests:

```text
partition: dgxh
constraint: h100&vram80g
gres: gpu:8
nodes: 1
ntasks: 1
cpus-per-task: 64
memory: 500G
walltime: 01:00:00
```

The 80GB constraint is intentional. Phase 0 live checks showed
`--constraint=h100` alone could target `dgxh-1` with `gpu:h100-40g:16`, which is
not the intended competition-class node.

## Default Candidate Order

The one-hour-oriented default is:

```text
dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384
```

Reasoning:

- `dense_sp8192_smoke` checks the base stack, CaseOps `sp8192`, FA3, fused
  kernels, GPTQ/package path, and parser.
- `qmlp_sp8192_smoke` checks the same-vocab qMLP path inside the full record
  stack.
- `qmlp_sp16384` is the main record-attempt candidate carried forward from
  Goal 2.

The order can be overridden at submission time with:

```bash
GOAL3_CANDIDATES="dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384" \
sbatch goal-3/h100-record-runner.sbatch
```

Full `qmlp_sp8192` can be inserted only if the reviewed allocation has enough
time or if the user chooses same-vocab measurement over the `sp16384` record
attempt.

## Outputs

Each run writes to:

```text
/nfs/hpc/share/peterj29/pg/goal-3-runs/<job-name>-<job-id>/
```

Expected files:

- `context.txt`;
- `gpu.txt`;
- `python.txt`;
- `env-smoke.json`;
- `git-status.txt`;
- `git-diff.stat`;
- `git-diff.patch`;
- `scratch-stage.txt`;
- `run-order.txt`;
- `progress.txt`;
- `candidates/<candidate>/seed_<seed>/stdout.log`;
- `candidates/<candidate>/seed_<seed>/stderr.log`;
- `candidates/<candidate>/seed_<seed>/env.txt`;
- `candidates/<candidate>/seed_<seed>/summary.json`;
- `candidates/<candidate>/seed_<seed>/status.json`;
- `final-status.json`.

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
bash -n goal-3/h100-record-runner.sbatch
bash -n goal-3/h100-repair-agent.sbatch
bash -n goal-3/scripts/run_candidate.sh
python3 -m py_compile goal-3/scripts/parse_train_log.py
python3 -m py_compile goal-3/scripts/env_smoke.py
```

Runtime checks pending:

- `srun --test-only` has not been refreshed after these scripts were created;
- no H100 script has been submitted;
- no final runner output exists yet.

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

- Final runner exists: complete.
- Runner has explicit resource request: complete.
- Runner has bounded candidate order: complete.
- Runner writes machine-readable final status: complete.
- Repair agent exists and is disabled by default: complete.
- H100 dry-run is current: pending.
- User approval for exact H100 request: pending.

## Next Phase

Phase 7 is the human approval gate. Before submitting H100 work, refresh live
Slurm state, run `srun --test-only` with the exact resource request, show the
script path and candidate order to the user, and wait for explicit approval.
