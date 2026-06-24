# Goal 3 Handoff: Check Submitted H100 Campaign

Last updated: 2026-06-23 20:36 America/Los_Angeles

This file is a self-contained handoff for a fresh Codex session. Start here if
the task is to check on the submitted Goal 3 H100 campaign.

## Current State

The approved autonomous H100 campaign has been submitted.

```text
Slurm job ID: 20487886
Script: goal-3/h100-campaign-runner.sbatch
Partition: dgxh
Constraint: h100&vram80g
GRES: gpu:8
Nodes: 1
Tasks: 1
CPUs per task: 64
Memory: 500G
Walltime: 06:00:00
Latest known state: PD, reason Priority
```

Fresh dry-run immediately before submission:

```text
Dry-run job: 20487885
Predicted start: 2026-06-28T08:29:30
Predicted node: dgxh-3
```

Expected run directory once the job starts:

```text
/nfs/hpc/share/peterj29/pg/goal-3-runs/goal3-h100-campaign-20487886
```

Remote project checkout:

```text
/nfs/hpc/share/peterj29/pg/src/pg
```

## Access Path

Local Mac to HPC submit node uses SSH ProxyJump through the Engineering gateway:

```bash
ssh -J peterj29@access.engr.oregonstate.edu peterj29@submit-a.hpc.engr.oregonstate.edu
```

For one-off checks from the local repo:

```bash
zsh -lic 'ssh -J peterj29@access.engr.oregonstate.edu peterj29@submit-a.hpc.engr.oregonstate.edu "cd /nfs/hpc/share/peterj29/pg/src/pg && squeue -j 20487886 -o \"%.18i %.12P %.32j %.8u %.2t %.12M %.10l %.6D %R\""'
```

Do not use the submit node for training, compilation, GPU diagnostics, dataset
export, or heavy filesystem work. Use it only for lightweight inspection,
Slurm status, logs, and static checks.

## What The Job Will Do

The campaign is deterministic and should run without intervention once Slurm
starts it.

Order:

1. Create the run directory.
2. Prepare runtime scratch/cache paths under:

```text
/scratch/$USER/$SLURM_JOB_ID/goal3
```

3. Activate:

```text
/nfs/hpc/share/peterj29/pg/envs/goal3-cu128
```

4. Validate/build runtime requirements, including FA3 import/install if needed.
5. Record context, Git state, module list, GPU inventory, Python deps.
6. Run `goal-3/scripts/env_smoke.py`.
7. Stage Goal 3 source plus CaseOps `sp8192` and `sp16384` data/tokenizers to
   node-local scratch.
8. Run short smoke gate:

```text
dense_sp8192_smoke seed 42
qmlp_sp8192_smoke seed 42
qmlp_sp16384_smoke seed 42
qmlp_sp16384_ttt_smoke seed 42
```

9. If smokes pass, run full dense baseline:

```text
dense_sp8192 seed 42
```

10. If the baseline hard gate passes, run full qMLP:

```text
qmlp_sp16384 seed 42
qmlp_sp16384 seed 0
qmlp_sp16384 seed 1234
```

11. Write summaries, artifact manifests, and `final-status.json`.

## Stop Gates

The job should stop early if:

- prepared Python env is missing;
- FA3 import/install fails;
- not exactly 8 H100 CUDA devices are visible;
- `lrzip` is missing or cannot execute on the allocated node;
- either CaseOps tokenizer fails expected vocab-size load;
- scratch staging fails;
- any smoke exits nonzero;
- any smoke reports an over-budget artifact;
- dense baseline exits nonzero;
- dense baseline artifact accounting is missing or over 16 MB;
- dense baseline post-TTT BPB is worse than `1.075`;
- dense baseline final train step count is below `4000`;
- a qMLP candidate exits nonzero or is over budget.

Strict dense parity target is also recorded but should not stop the run by
itself:

```text
BPB <= 1.065
steps >= 4500
```

Hard dense gate:

```text
BPB <= 1.075
steps >= 4000
exit_code = 0
artifact_under_limit = true
```

## Files To Inspect

If the job is still pending:

```bash
squeue -j 20487886 -o "%.18i %.12P %.32j %.8u %.2t %.12M %.10l %.6D %R"
```

If the job is running or finished:

```bash
RUN=/nfs/hpc/share/peterj29/pg/goal-3-runs/goal3-h100-campaign-20487886
ls -la "$RUN"
cat "$RUN/progress.txt" 2>/dev/null || true
cat "$RUN/final-status.json" 2>/dev/null || true
cat "$RUN/smoke-gate.json" 2>/dev/null || true
cat "$RUN/baseline-parity.json" 2>/dev/null || true
```

Expected run-level files:

```text
context.txt
gpu.txt
python.txt
runtime-storage.txt
runtime-setup/imports-before.json
runtime-setup/imports-after.json
runtime-setup/fa3-install.txt, if runtime install ran
env-smoke.json
git-status.txt
git-diff.stat
git-diff.patch
source-snapshot/goal-3/
source-snapshot.sha256
scratch-stage.txt
run-order.txt
progress.txt
smoke-gate.json
baseline-parity.json
final-status.json
```

Expected candidate directories:

```text
candidates/<candidate>/seed_<seed>/stdout.log
candidates/<candidate>/seed_<seed>/stderr.log
candidates/<candidate>/seed_<seed>/env.txt
candidates/<candidate>/seed_<seed>/summary.json
candidates/<candidate>/seed_<seed>/status.json
candidates/<candidate>/seed_<seed>/artifacts.json
```

Useful status checks after terminal state:

```bash
sacct -j 20487886 --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End,NodeList%40
```

## How To Interpret Outcomes

If `final-status.json` has `"status": "passed"`:

- parse qMLP seed summaries;
- compare qMLP post-TTT BPB against:

```text
Primary record target: 1.06108 mean post-TTT BPB
Fallback/compliance target: 1.06141 mean post-TTT BPB
```

- verify artifact size is under `16,000,000` bytes for every completed final
  candidate;
- update `goal-3/findings-summary.md`, `goal-3/status.md`, and
  `goal-3/jobs.csv`.

If `final-status.json` has `"status": "failed"`:

- identify the first failed gate from `progress.txt`, `smoke-gate.json`,
  `baseline-parity.json`, and candidate `status.json`;
- do not submit another H100 job automatically;
- summarize the failure and ask the user before retrying.

If there is no `final-status.json` but the job is terminal:

- inspect Slurm stdout/stderr:

```text
goal-3/logs/goal3-h100-campaign-20487886.out
goal-3/logs/goal3-h100-campaign-20487886.err
```

- inspect the run directory if it exists;
- update docs with the missing-final-status failure.

## Prebuilt Inputs

Already prepared before submission:

```text
sp8192 data:
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved

sp8192 tokenizer:
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model

sp16384 data:
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/datasets/datasets/fineweb10B_sp16384_lossless_caps_caseops_v1_reserved

sp16384 tokenizer:
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/datasets/tokenizers/fineweb_16384_bpe_lossless_caps_caseops_v1_reserved.model

Python env:
/nfs/hpc/share/peterj29/pg/envs/goal3-cu128

lrzip:
/nfs/hpc/share/peterj29/pg/tools/lrzip/bin/lrzip
```

The two CaseOps exports are about `3.2G` total. Live pre-submit storage check
showed about `1.4T` free on `/nfs/hpc/share/peterj29`.

## Important Constraints

- Do not submit another H100/H200 job without explicit user approval.
- Do not cancel job `20487886` unless the user explicitly asks or there is a
  clear safety issue.
- Do not run training, compilation, tokenizer export, GPU commands, or heavy
  filesystem scans on the submit node.
- Do not broaden into sweeps. The submitted campaign is the bounded approved
  run.
- Facts from Slurm/logs beat stale docs.

## Docs To Update After Checking

After checking the job, update:

```text
goal-3/jobs.csv
goal-3/status.md
goal-3/findings-summary.md
goal-3/8-campaign.md
```

If the job has produced interpretable results, summarize:

- Slurm terminal state and exit code;
- first failed gate or successful completion;
- dense baseline metrics;
- qMLP seed metrics;
- artifact sizes;
- comparison to `1.06108` and `1.06141`;
- whether another H100 run is justified and why.
