# Goal 3 Status

Last updated: 2026-06-23 16:38 America/Los_Angeles

## Current Phase

Phase 7: Human review and H100 request gate.

Status: Phase 3 qMLP source port is statically complete, and the H100 runner
artifacts are scripted and `bash -n` checked. The runner records dirty diffs,
stages Goal 3 source/data inputs to node-local scratch, launches candidates
through Slurm-accounted `srun` by default, and writes machine-readable
`final-status.json` on normal completion or early failure when possible.
`goal-3/scripts/static_goal3_audit.py` now provides a repeatable local/remote
text-level guardrail check for qMLP integration and H100 runner invariants.
`goal-3/` has been synced to the remote HPC checkout through the OSU gateway
and remote submit-node static checks passed, including the static Goal 3 audit.
CPU Slurm prep jobs have completed for the shared Python
environment and user-local `lrzip`. Phase 7 approval packet exists at
`goal-3/7-approval.md`. No H100/H200 work has been submitted, and the
15-minute H100 env smoke is pending explicit approval.

## Objective

Prepare and execute a full 8xH100 Parameter Golf record-track attempt that
starts from the strongest known compliant competition stack and adds qMLP in a
controlled way.

Decision question:

```text
Can qMLP improve the best-under-16MB record-track result on the intended
8xH100 H100/FA3 competition setup?
```

## Required Reading Completed

- `goal-3/0-plan.md`
- `goal-3/0-loop.md`
- `goal-3/0-prompt.md`
- `goal-2/findings-summary.md`
- `osu-hpc-agent-guide/AGENTS.md`
- `osu-hpc-agent-guide/ssh-setup.md`
- Relevant sections of `osu-hpc-agent-guide/OSU_HPC_AGENT_GUIDE.md`
- Primary base README:
  `parameter-golf/records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/README.md`
- Fallback README:
  `parameter-golf/records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/README.md`
- Primary base `train_gpt.py` structure scan around Hyperparameters, MLP,
  GPT forward/TTT paths, optimizer grouping, GPTQ/LQER, and serialization.

## Selected Base And Fallback

Primary base:

```text
parameter-golf/records/track_10min_16mb/
2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611
```

Reason: strongest local H100 record candidate, reporting 3-seed post-TTT mean
`1.06108` BPB, about `15.9 MB`, and about `4931` training steps in 600 seconds
on 8xH100 SXM 80GB.

Fallback:

```text
parameter-golf/records/track_10min_16mb/
2026-04-29_SmearGateBOSFix_3Seed_1.06141
```

Reason: slightly weaker but clearer compliance reproduction with
`GPTQ_RESERVE_SECONDS=8.0`, explicit GPTQ timing, and 3-seed post-TTT mean
`1.06141` BPB.

## Local Source State

Local workspace:

```text
path: /Users/jake/Developer/pg
branch: mac
HEAD: 60a7c1afbe43a47f4b8d9e1db8b35474eb5c2ece
dirty: yes
```

Observed local dirty state:

```text
M goal-3/6-runner.md
M goal-3/7-approval.md
M goal-3/findings-summary.md
M goal-3/h100-env-smoke.sbatch
M goal-3/h100-record-runner.sbatch
M goal-3/h100-repair-agent.sbatch
M goal-3/h100-short-smoke.sbatch
M goal-3/scripts/common.sh
M goal-3/scripts/run_candidate.sh
?? goal-3/scripts/static_goal3_audit.py
M goal-3/status.md
m parameter-golf
```

Submodules:

```text
f5c079314c4877fbb0af378c0abade5a8ca33d3a parameter-golf (heads/main)
fb7b546294aecdabace2f5fab0527001df320b77 qham (heads/master)
```

## Remote HPC Source State

Remote project checkout exists at:

```text
/nfs/hpc/share/peterj29/pg/src/pg
```

Remote checkout snapshot from 2026-06-23 14:53:

```text
branch: mac
HEAD: e45bef8afd9c4129850aae14f4d0c1fd8543fbad
dirty: yes
```

Important note: the remote checkout is stale relative to local Goal 3 work and
still contains older `goal/`/`goal-2/` state. Sync or patch the remote checkout
before any Goal 3 Slurm work.

Remote Goal 3 sync from 2026-06-23 16:38:

```text
goal-3/ synced to /nfs/hpc/share/peterj29/pg/src/pg/goal-3
remote static checks: passed
remote git status for Goal 3: ?? goal-3/
remote parameter-golf status: m parameter-golf
```

Remote static checks run:

```text
bash -n goal-3/scripts/common.sh
bash -n goal-3/scripts/run_candidate.sh
bash -n goal-3/prepare-env.sbatch
bash -n goal-3/h100-env-smoke.sbatch
bash -n goal-3/h100-short-smoke.sbatch
bash -n goal-3/h100-record-runner.sbatch
bash -n goal-3/h100-repair-agent.sbatch
python compile() syntax check for env_smoke.py, parse_train_log.py, train_gpt.py
python goal-3/scripts/static_goal3_audit.py
```

Transport note: local `hpc` alias was not available directly. The successful
sync/check path used SSH ProxyJump through:

```text
peterj29@access.engr.oregonstate.edu -> peterj29@submit-a.hpc.engr.oregonstate.edu
```

## Live HPC State

Read-only live check timestamp:

```text
submit host: submit-a.ib.coehpc
time: 2026-06-23T16:05:06-07:00
user: peterj29
association: coehpc|eecs|peterj29|||normal
current queue: no jobs listed for peterj29
```

Storage:

```text
/nfs/stak/users/peterj29: 25G total, 14G used, 12G available
/nfs/hpc/share/peterj29: 1.5T total, 156G used, 1.4T available
```

Visible QOS facts from live `sacctmgr`:

```text
dgxh MaxTRESPU: cpu=224, gres/gpu=8, mem=2000G
dgxh MaxTRESRunMinsPU: cpu=92160, gres/gpu=2880, mem=720T
```

H100 node facts from live `sinfo`:

```text
dgxh-1: gpu:h100-40g:16, features h100,vram40g
dgxh-2: gpu:8, drained, features h100,vram80g
dgxh-3: gpu:8, mixed, features h100,vram80g
dgxh-4: gpu:8, mixed, features h200,vram80g,vram140g
```

Dry-run findings:

```text
--constraint=h100:
  previously predicted dgxh-1
  not acceptable for the intended 8xH100 80GB run because dgxh-1 is h100-40g

goal-3/h100-env-smoke.sbatch equivalent:
  srun --test-only -p dgxh --constraint="h100&vram80g" --gres=gpu:8
  --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=500G --time=00:15:00 true
  predicted dgxh-3 at 2026-06-27T08:29:30

goal-3/h100-record-runner.sbatch equivalent:
  srun --test-only -p dgxh --constraint="h100&vram80g" --gres=gpu:8
  --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=500G --time=01:00:00 true
  predicted dgxh-3 at 2026-06-27T08:29:30
```

## Current Decision

Use the 2026-04-27 record as the primary base. Require the H100 80GB constraint
for all final Goal 3 H100 scripts, currently:

```bash
--partition=dgxh --constraint="h100&vram80g" --gres=gpu:8
```

Do not submit H100/H200 jobs until the final runner exists, passes static and
non-scarce checks, and the user approves the exact request.

## Phase 1 Result

`goal-3/1-base.md` maps the base stack and qMLP implementation surfaces.

Key implementation decision:

- Use the banked qMLP pattern from the 2026-04-23 record stack, not the simpler
  `QuaternionLinear` path in `parameter-golf/train_gpt.py`.
- The 2026-04-27 primary base has newer per-group `lrzip` compression, so qMLP
  component names must be added to GPTQ/unbank/rebank and compression handling.
- `GPTQ_RESERVE_SECONDS` must be reviewed before the final run because 04-27
  used `0.5` while the 04-29 compliance reproduction demonstrated the safer
  `8.0` setting.

## Phase 2 Result

`goal-3/2-data.md` and `goal-3/data-manifest.csv` record the required CaseOps
data manifests.

Ready data paths:

```text
sp8192 DATA_PATH:
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved

sp8192 TOKENIZER_PATH:
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model

sp16384 DATA_PATH:
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/datasets/datasets/fineweb10B_sp16384_lossless_caps_caseops_v1_reserved

sp16384 TOKENIZER_PATH:
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/datasets/tokenizers/fineweb_16384_bpe_lossless_caps_caseops_v1_reserved.model
```

Both have 80 train shards and validation token/byte sidecar files. Direct
SentencePiece vocab-size loading is deferred to the environment smoke because
the submit-node system Python lacks `sentencepiece`.

## Next Action

Ask the user to approve only the 15-minute H100 env smoke documented in
`goal-3/7-approval.md`. Do not submit the env smoke, short smoke, or record
runner until the exact corresponding request is explicitly approved.

Phase 3 preserved `QUAT_MLP=0` behavior while adding:

- `QUAT_MLP` / `QUAT_MLP_IMPL`;
- banked qMLP MLP weights;
- qMLP GPTQ Hessians;
- qMLP unbank/rebank;
- qMLP component compression handling;

## Phase 3 Result

`goal-3/3-qmlp-port.md` documents the staged qMLP implementation.

Staged source:

```text
goal-3/stage/primary-qmlp/train_gpt.py
```

Static verification:

```text
python3 -m py_compile goal-3/stage/primary-qmlp/train_gpt.py: passed
```

Runtime qMLP smoke is still pending.

## Script Artifacts

Created and statically checked:

```text
goal-3/prepare-env.sbatch
goal-3/h100-env-smoke.sbatch
goal-3/h100-short-smoke.sbatch
goal-3/h100-record-runner.sbatch
goal-3/h100-repair-agent.sbatch
goal-3/scripts/common.sh
goal-3/scripts/env_smoke.py
goal-3/scripts/run_candidate.sh
goal-3/scripts/parse_train_log.py
```

Default one-hour runner order:

```text
dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384
```

H100 approval status: env-smoke request prepared in `goal-3/7-approval.md`;
not yet granted.

Additional prep now complete:

- `goal-3/compliance-note.md` states the qMLP compliance assumptions and the
  runtime proof still required.
- CPU environment prep completed as job `20487397`, producing
  `/nfs/hpc/share/peterj29/pg/envs/goal3-cu128`.
- CPU tools prep completed as job `20487617`, producing
  `/nfs/hpc/share/peterj29/pg/tools/lrzip/bin/lrzip` with user-local LZO and
  LZ4 libraries. Direct submit-node execution of that binary fails due the
  submit node's older glibc, so the valid runtime check is the H100 env smoke.
- `goal-3/scripts/env_smoke.py` checks CUDA device count, FA3 import, `lrzip`
  presence and executability, and tokenizer vocab sizes.
- `goal-3/scripts/parse_train_log.py` now reports both the last periodic
  train-loss step and the final wallclock stop step as `train_steps_final`.
- `goal-3/scripts/static_goal3_audit.py` checks qMLP flag/component wiring,
  qMLP Hessian and compression key coverage, candidate-to-vocab mappings,
  H100 80GB constraints, final-status traps, and bounded runner defaults.
  It intentionally avoids newer Python-only syntax because the submit-node
  `python3` used for static checks is older than the local Mac Python.
- `goal-3/scripts/run_candidate.sh` launches `torchrun` through `srun` by
  default inside Slurm allocations and records the effective launcher in each
  candidate `env.txt`.
- `goal-3/h100-record-runner.sbatch` now bounds the default one-hour candidate
  timeouts to `8m + 8m + 36m`; `goal-3/h100-short-smoke.sbatch` now defaults
  to `10m` per smoke candidate inside its 45-minute allocation.
- Candidate `status.json` now records `timeout` and `timed_out` so a timeout
  kill is explicit in machine-readable summaries.
- H100 env, short-smoke, record-runner, and optional repair-agent scripts now
  use `goal3_write_final_status` traps so early failures still leave
  `final-status.json` when possible.
- H100 short-smoke and record-runner normal summaries include candidate order,
  seed, timeout settings, and per-candidate `status.json` contents.
- `goal-3/scripts/common.sh` exports required Goal 3 paths for child processes.
- `goal-3/scripts/common.sh` records `git-status.txt`, `git-diff.stat`, and
  `git-diff.patch`.
- `goal-3/h100-short-smoke.sbatch` and `goal-3/h100-record-runner.sbatch` stage
  the Goal 3 source tree and both required CaseOps data/tokenizer sets to
  `/scratch/$USER/$SLURM_JOB_ID/goal3`.
