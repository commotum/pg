# Phase 7: Approval

## Overview

This phase is the human review gate before any scarce H100/H200 submission. The
CPU preparation is complete, the H100 scripts exist, and the current safe next
step is an H100 environment smoke. No H100/H200 job has been submitted.

The immediate approval request is for the environment smoke only. It does not
approve the later one-hour record runner.

## Live Slurm Check

Live check timestamp:

```text
submit host: submit-a.ib.coehpc
time: 2026-06-23T16:05:06-07:00
user queue: empty
association: coehpc|eecs|peterj29|||normal
```

Relevant H-class node state:

```text
dgxh-1: mixed, gpu:h100-40g:16, features h100,vram40g
dgxh-2: drained, gpu:8, features h100,vram80g
dgxh-3: mixed, gpu:8, features h100,vram80g
dgxh-4: mixed, gpu:8, features h200,vram80g,vram140g
```

Relevant visible QOS fact:

```text
dgxh MaxTRESPU: cpu=224, gres/gpu=8, mem=2000G
dgxh MaxTRESRunMinsPU: cpu=92160, gres/gpu=2880, mem=720T
```

The `h100&vram80g` constraint is still required. Plain `h100` can match
`dgxh-1`, which is currently advertised as `h100-40g`.

## Dry-Run Evidence

H100 env smoke dry-run:

```bash
srun --test-only -p dgxh --constraint="h100&vram80g" --gres=gpu:8 \
  --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=500G \
  --time=00:15:00 true
```

Result:

```text
srun: Job 20487620 to start at 2026-06-27T08:29:30 a using 64 processors on nodes dgxh-3 in partition dgxh
```

Future one-hour record runner dry-run:

```bash
srun --test-only -p dgxh --constraint="h100&vram80g" --gres=gpu:8 \
  --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=500G \
  --time=01:00:00 true
```

Result:

```text
srun: Job 20487621 to start at 2026-06-27T08:29:30 a using 64 processors on nodes dgxh-3 in partition dgxh
```

These are scheduler fit checks only. They did not submit H100 work.

## Immediate Approval Request

Script path:

```text
goal-3/h100-env-smoke.sbatch
```

Submission command, after approval:

```bash
cd /nfs/hpc/share/peterj29/pg/src/pg
sbatch --parsable goal-3/h100-env-smoke.sbatch
```

Requested resources:

```text
partition: dgxh
constraint: h100&vram80g
GRES: gpu:8
nodes: 1
ntasks: 1
cpus-per-task: 64
memory: 500G
walltime: 00:15:00
stdout: goal-3/logs/%x-%j.out
stderr: goal-3/logs/%x-%j.err
```

Expected maximum scarce compute consumed:

```text
15 wallclock minutes on one 8xH100 80GB node
120 H100-GPU-minutes
```

Run directory:

```text
/nfs/hpc/share/peterj29/pg/goal-3-runs/goal3-h100-env-$SLURM_JOB_ID
```

What it does:

1. Activates `/nfs/hpc/share/peterj29/pg/envs/goal3-cu128`.
2. Records host, Slurm environment, Git state, module list, GPU inventory, and
   Python dependency context.
3. Runs `goal-3/scripts/env_smoke.py` under `srun`.
4. Verifies 8 visible CUDA devices.
5. Verifies `torch`, `triton`, `sentencepiece`, `brotli`, and
   `flash_attn_interface` imports.
6. Verifies `lrzip` is on `PATH` and can execute a lightweight version/help
   probe.
7. Verifies `sp8192` and `sp16384` tokenizer vocab sizes.
8. Writes `env-smoke.json` and `final-status.json`.
9. Uses an exit trap so early failure still writes `final-status.json` when
   possible.

What it does not do:

- no training;
- no tokenizer export;
- no package smoke;
- no qMLP candidate run;
- no Codex-on-node repair agent.

Stop conditions:

- missing Python environment;
- fewer or more than 8 visible CUDA devices;
- FA3 import failure;
- `lrzip` missing or not runnable on the H100 node;
- tokenizer load failure or wrong vocab size;
- any nonzero script exit, with `final-status.json` written when possible.

Known risks:

- The H100 node may not become available until the predicted Slurm start time,
  currently `2026-06-27T08:29:30`.
- The compute-built `lrzip` binary cannot run on the submit node because of an
  older submit-node glibc; the H100 env smoke is the meaningful check.
- FA3 was installed into the shared Python env, but the submit node cannot
  validate it for the same glibc/H100-runtime reasons.
- Passing this smoke proves environment viability only; it does not prove qMLP
  package size, runtime, or BPB.

Approval needed:

```text
Approve submitting goal-3/h100-env-smoke.sbatch with the resources above.
```

## Later Record Runner Review

The later one-hour runner is already scripted but should require a separate
approval after the env smoke passes.

Script path:

```text
goal-3/h100-record-runner.sbatch
```

Requested resources:

```text
partition: dgxh
constraint: h100&vram80g
GRES: gpu:8
nodes: 1
ntasks: 1
cpus-per-task: 64
memory: 500G
walltime: 01:00:00
stdout: goal-3/logs/%x-%j.out
stderr: goal-3/logs/%x-%j.err
```

Default candidate order:

```text
dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384
```

Default candidate timeouts:

```text
dense_sp8192_smoke: 8m
qmlp_sp8192_smoke: 8m
qmlp_sp16384: 36m
```

The default candidate-timeout ceiling is therefore 52 minutes inside the
one-hour allocation, leaving room for environment smoke, context capture,
scratch staging, parser summaries, and final status writing.

Default seed:

```text
42
```

Candidate launch behavior:

```text
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate `env.txt` records the effective launcher. Direct non-`srun`
candidate launch is available only by setting `GOAL3_USE_SRUN_LAUNCHER=0`, which
should require review before H100 use.

Runner stop conditions:

- env smoke failure;
- scratch staging failure;
- unknown candidate;
- candidate nonzero exit;
- parsed artifact size at or above `16,000,000` bytes;
- missing required data/tokenizer files.

This runner approval is not currently requested. The correct next step is the
15-minute H100 env smoke.

Normal runner output:

- `final-status.json` includes candidate order, seed, smoke/full timeout
  settings, per-candidate `status.json` payloads, and parsed summaries.
- The exit trap writes a fallback `final-status.json` for early failures before
  the normal summary path is reached.

## Verification Steps

Completed before this document:

```bash
bash -n goal-3/h100-env-smoke.sbatch
bash -n goal-3/h100-short-smoke.sbatch
bash -n goal-3/h100-record-runner.sbatch
bash -n goal-3/h100-repair-agent.sbatch
python3 -m py_compile goal-3/scripts/env_smoke.py
python3 -m py_compile goal-3/scripts/parse_train_log.py
python3 -m py_compile goal-3/scripts/static_goal3_audit.py
python3 -m py_compile goal-3/stage/primary-qmlp/train_gpt.py
python3 goal-3/scripts/static_goal3_audit.py
```

The H100 sbatch syntax checks were rerun locally on 2026-06-23 at 16:30 Pacific
after moving the short-smoke and record-runner final-status traps ahead of the
env smoke and after adding richer normal final summaries.

The static Goal 3 qMLP/runner audit passed locally on 2026-06-23 at 16:35
Pacific. It checks qMLP wiring, candidate mappings, H100 80GB constraints,
final-status traps, scratch staging, and default candidate timeout/order
guardrails without importing H100-only dependencies.

Remote static checks also passed after syncing `goal-3/` to:

```text
/nfs/hpc/share/peterj29/pg/src/pg/goal-3
```

The latest sync/check was completed on 2026-06-23 at 16:38 Pacific through the
OSU gateway using SSH ProxyJump to `submit-a.hpc.engr.oregonstate.edu`. Remote
static checks included `python3 goal-3/scripts/static_goal3_audit.py`, which
passed.

## Completion Requirements

- Live Slurm state recorded: complete.
- Exact env-smoke script and resources documented: complete.
- Exact dry-run estimate recorded: complete.
- Known risks and stop conditions documented: complete.
- User explicitly approves H100 env-smoke submission: pending.
- H100 env-smoke job submitted and tracked in `goal-3/jobs.csv`: pending.

## Findings

The H100 80GB request remains schedulable through `dgxh-3`, but not immediate.
The current scheduler prediction for both the 15-minute env smoke and a later
one-hour record runner is `2026-06-27T08:29:30`.

The next action is to ask for approval to submit only
`goal-3/h100-env-smoke.sbatch`.
