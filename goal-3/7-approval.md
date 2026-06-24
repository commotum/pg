# Phase 7: Approval

## Overview

This phase is the human review gate before any scarce H100/H200 submission. No
H100/H200 job has been submitted.

The approval target has been revised. Do not ask for a separate 15-minute
environment smoke followed by a later full runner. That creates multiple queue
waits for one campaign. The next H100 request should be a single autonomous
campaign allocation that validates the environment, runs short dense/qMLP
candidate smokes, checks dense baseline validity, and then runs the qMLP
contender seeds if and only if the hard baseline gate passes.

## Approval Target

Script path:

```text
goal-3/h100-campaign-runner.sbatch
```

Submission command, after explicit approval:

```bash
cd /nfs/hpc/share/peterj29/pg/src/pg
sbatch --parsable goal-3/h100-campaign-runner.sbatch
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
walltime: 06:00:00
stdout: goal-3/logs/%x-%j.out
stderr: goal-3/logs/%x-%j.err
```

Expected maximum scarce compute consumed:

```text
360 wallclock minutes on one 8xH100 80GB node
2880 H100-GPU-minutes
```

Run directory:

```text
/nfs/hpc/share/peterj29/pg/goal-3-runs/goal3-h100-campaign-$SLURM_JOB_ID
```

## Campaign Sequence

The campaign runner does this in order:

1. Activates `/nfs/hpc/share/peterj29/pg/envs/goal3-cu128`.
2. Runs `goal3_ensure_runtime_requirements`, which records import status and
   attempts FA3 install from the documented wheel source if
   `flash_attn_interface` is missing and runtime install is enabled.
3. Records host, Slurm environment, Git state, module list, GPU inventory, and
   Python dependency context.
4. Runs `goal-3/scripts/env_smoke.py` under `srun`.
5. Stages Goal 3 source plus `sp8192` and `sp16384` CaseOps data/tokenizers to
   `/scratch/$USER/$SLURM_JOB_ID/goal3` when scratch staging is enabled.
6. Runs short distributed candidate smokes:
   - `dense_sp8192_smoke`, seed 42;
   - `qmlp_sp8192_smoke`, seed 42;
   - `qmlp_sp16384_smoke`, seed 42;
   - `qmlp_sp16384_ttt_smoke`, seed 42.
7. Writes `smoke-gate.json` and stops with exit code `60` if any smoke exits
   nonzero or produces an over-budget artifact when artifact accounting is
   available. If one smoke fails, later smoke candidates are marked
   `not_attempted` rather than spending more allocation time.
8. Runs exact dense/base `sp8192`, seed 42, as a full baseline parity run.
9. Writes `baseline-parity.json` even when the baseline candidate exits
   nonzero; stops with exit code `70` only if the baseline fails the hard
   validity gate. A borderline baseline records `strict_parity_passed=false`
   but still allows the predeclared qMLP seeds to run so the allocation is not
   wasted on an over-tight parity threshold.
10. Runs qMLP `sp16384` full candidates for seeds `42`, `0`, and `1234`.
11. Writes per-candidate logs, summaries, status files, artifact manifests, and a
   final campaign summary with env-smoke, smoke-gate, baseline-parity, qMLP
   mean/std, candidate artifacts, and source-snapshot pointers when all qMLP
   seeds complete.

Default smoke gate:

```text
GOAL3_SMOKE_CANDIDATES="dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384_smoke qmlp_sp16384_ttt_smoke"
GOAL3_SMOKE_SEED=42
GOAL3_SMOKE_TIMEOUT=20m
```

Default baseline gates:

```text
strict parity target:
  GOAL3_BASELINE_PARITY_MAX_BPB=1.065
  GOAL3_BASELINE_PARITY_MIN_STEPS=4500

hard stop gate:
  GOAL3_BASELINE_HARD_MAX_BPB=1.075
  GOAL3_BASELINE_HARD_MIN_STEPS=4000

artifact_under_limit=true
exit_code=0
```

Default full-run timeout:

```text
GOAL3_FULL_TIMEOUT=120m
```

That timeout is intentionally much larger than the 10-minute training budget
because the record stack still needs post-training quantization, TTT/eval,
compression, and stage-out time. It is a backstop for a stuck candidate, not an
expected runtime target. Training itself remains bounded by the record script's
`MAX_WALLCLOCK_SECONDS=600` default.

## Required Candidate Runs

The reviewed campaign default is:

| Order | Candidate | Seed | Purpose |
|---:|---|---:|---|
| 1 | `dense_sp8192_smoke` | 42 | check dense record-stack launch/package path cheaply |
| 2 | `qmlp_sp8192_smoke` | 42 | check same-vocab qMLP wiring cheaply |
| 3 | `qmlp_sp16384_smoke` | 42 | check carry-forward qMLP vocab/tokenizer path cheaply |
| 4 | `qmlp_sp16384_ttt_smoke` | 42 | cheaply exercise qMLP plus TTT LoRA hook path |
| 5 | `dense_sp8192` | 42 | prove OSU H100 setup can reproduce the known record path closely enough |
| 6 | `qmlp_sp16384` | 42 | first qMLP contender after baseline hard validity passes |
| 7 | `qmlp_sp16384` | 0 | qMLP seed replication |
| 8 | `qmlp_sp16384` | 1234 | qMLP seed replication and direct record seed-set comparison |

## Stop Conditions

The runner must stop before qMLP full runs if:

- the prepared Python environment is missing;
- FA3 is missing and runtime install/build fails;
- fewer or more than 8 H100 CUDA devices are visible;
- `lrzip` is missing or cannot execute on the allocated node;
- either CaseOps tokenizer fails to load with the expected vocab size;
- scratch staging fails;
- any short candidate smoke exits nonzero;
- any short candidate smoke reports an over-budget artifact;
- dense/base baseline exits nonzero;
- dense/base baseline artifact accounting is missing or over 16 MB;
- dense/base baseline post-TTT BPB is worse than the hard stop threshold
  `1.075`;
- dense/base baseline final train step count is below the hard stop threshold
  `4000`.

The runner must stop during qMLP if:

- a qMLP candidate exits nonzero;
- qMLP artifact accounting is missing or over 16 MB;
- any compliance-sensitive check fails.

## Outputs To Preserve

Expected run files include:

```text
context.txt
gpu.txt
python.txt
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
candidates/<candidate>/seed_<seed>/stdout.log
candidates/<candidate>/seed_<seed>/stderr.log
candidates/<candidate>/seed_<seed>/env.txt
candidates/<candidate>/seed_<seed>/summary.json
candidates/<candidate>/seed_<seed>/status.json
candidates/<candidate>/seed_<seed>/artifacts.json
final-status.json
```

The artifact manifest hashes every non-log file in each candidate directory.
Full model artifacts, quantized submission artifacts, parser summaries, and
hashes must remain in shared storage after stage-out.

If the campaign exits early, `final-status.json` is still campaign-aware: it
includes any available `env-smoke.json`, `baseline-parity.json`, per-candidate
`summary.json`, `status.json`, `artifacts.json`, smoke-gate status, and source
snapshot manifest paths that were written before failure.

## Live Slurm Check

Last recorded live check:

```text
submit host: submit-a.ib.coehpc
time: 2026-06-23T18:05:00-07:00
user queue: empty
association: coehpc|eecs|peterj29|||normal
```

Relevant H-class node state:

```text
dgxh-1: mixed-, gpu:h100-40g:16, features h100,vram40g
dgxh-2: drained*, gpu:8, features h100,vram80g
dgxh-3: mixed-, gpu:8, features h100,vram80g
dgxh-4: mixed-, gpu:8, features h200,vram80g,vram140g
```

Relevant visible QOS fact:

```text
dgxh MaxTRESPU: cpu=224, gres/gpu=8, mem=2000G
dgxh MaxTRESRunMinsPU: cpu=92160, gres/gpu=2880, mem=720T
```

The `h100&vram80g` constraint is still required. Plain `h100` can match
`dgxh-1`, which is advertised as `h100-40g`.

## Dry-Run Evidence

Exact campaign dry-run:

```bash
srun --test-only -p dgxh --constraint="h100&vram80g" --gres=gpu:8 \
  --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=500G \
  --time=06:00:00 true
```

Result:

```text
srun: Job 20487754 to start at 2026-06-27T20:29:30 using 64 processors on
nodes dgxh-3 in partition dgxh
```

This is a scheduler fit check only. It did not submit H100 work.

Rejected longer-padding check, run only to confirm whether more than six hours
was schedulable and not submitted as a job:

```bash
srun --test-only -p dgxh --constraint="h100&vram80g" --gres=gpu:8 \
  --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=500G \
  --time=08:00:00 true
```

Result:

```text
srun: error: MaxGRESRunMinsPerUser
allocation failure: Job violates accounting/QOS policy
```

Interpretation: the live `dgxh` QOS exposes `gres/gpu=2880` run-minutes per
user. For 8 GPUs, that is exactly `360` wallclock minutes, so the six-hour
request is the maximum visible 8xH100 request rather than an arbitrary cap.

## Known Risks

- The current scheduler prediction is only a prediction. It can move before
  submission as queue state changes.
- FA3 was installed into the shared Python env, but the runner still includes a
  runtime install path because H100/runtime validation is the meaningful check.
- The compute-built `lrzip` binary cannot run on the submit node because of an
  older submit-node glibc; the campaign validates it on the allocated node.
- The dense baseline may miss the strict parity target because OSU
  software/hardware differs from the record author's setup. That is recorded as
  a comparison caveat, but it no longer stops qMLP unless the hard validity gate
  fails.
- Three qMLP full seeds may not all fit if setup, baseline, TTT, quantization,
  or compression is slower than expected. The logs and final status should make
  partial progress interpretable.

## Verification Steps

Local checks to rerun after any campaign edit:

```bash
bash -n goal-3/h100-campaign-runner.sbatch
bash -n goal-3/scripts/common.sh
bash -n goal-3/scripts/run_candidate.sh
python3 -m py_compile goal-3/scripts/env_smoke.py
python3 -m py_compile goal-3/scripts/parse_train_log.py
python3 -m py_compile goal-3/scripts/static_goal3_audit.py
python3 -m py_compile goal-3/stage/primary-qmlp/train_gpt.py
python3 goal-3/scripts/static_goal3_audit.py
```

Remote submit-node static checks must pass after syncing `goal-3/` to:

```text
/nfs/hpc/share/peterj29/pg/src/pg/goal-3
```

## Completion Requirements

- Campaign runner exists and passes local static checks: complete.
- Approval packet names the campaign runner, not the old env-smoke-only job:
  complete.
- Exact six-hour dry-run estimate recorded: complete.
- Remote static checks pass after the latest qMLP+TTT smoke/default sync:
  complete.
- User explicitly approves `goal-3/h100-campaign-runner.sbatch`: pending.
- H100 campaign job submitted and tracked in `goal-3/jobs.csv`: pending.

## Findings

The approval target is now the autonomous six-hour campaign runner. The
15-minute env smoke and one-hour record runner remain useful components and
fallback scripts, but they are not the recommended next H100 submission.
