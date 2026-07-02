# Goal 3 Status

Last updated: 2026-07-02 00:08 America/Los_Angeles

## Current Phase

Phase 8: fixed H100 campaign completed; result analysis complete.

Status: The fixed six-hour `goal-3/h100-campaign-runner.sbatch` request ran as
Slurm job `20517007`, completed successfully, and wrote
`final-status.json` with `"status": "passed"` and exit code `0`.

Slurm terminal state:

```text
JobID: 20517007
State: COMPLETED
ExitCode: 0:0
Node: dgxh-3
Start: 2026-07-01T07:32:15
End: 2026-07-01T10:02:57
Elapsed: 02:30:42
```

The ETA checked on 2026-06-29 was `2026-07-01T09:32:55`, so the job started
about 2 hours earlier than that ETA. It completed about 2.5 hours after actual
start.

Run directory:

```text
/nfs/hpc/share/peterj29/pg/goal-3-runs/goal3-h100-campaign-20517007
```

The environment smoke passed on 8x NVIDIA H100 80GB HBM3 with
`torch==2.9.1+cu128`, FA3 importable, both CaseOps tokenizers loading expected
vocab sizes, and `lrzip` runnable on the allocated node.

The smoke gate passed for `dense_sp8192_smoke`, `qmlp_sp8192_smoke`,
`qmlp_sp16384_smoke`, and `qmlp_sp16384_ttt_smoke`.

Dense baseline gate:

```text
candidate: dense_sp8192 seed 42
post-TTT BPB: 1.06843496
train steps: 4921
submission bytes: 15,907,532
hard gate: passed
strict parity target: failed warning only, BPB > 1.065
```

qMLP `sp16384` final seeds:

```text
seed 42:   post-TTT BPB 1.12930396, steps 4718, bytes 10,546,381
seed 0:    post-TTT BPB 1.12989323, steps 4717, bytes 10,547,376
seed 1234: post-TTT BPB 1.13081377, steps 4717, bytes 10,549,448
mean: 1.1300036533
stdev: 0.0007609379
```

Interpretation: qMLP stayed comfortably under the 16 MB artifact budget but did
not improve the dense baseline or the primary/fallback record targets. The
answer to the Goal 3 decision question is no for this implemented qMLP
candidate on the intended 8xH100 setup.

The previous campaign job `20487886` failed before training. `sacct` shows
`FAILED`, exit `1:0`, elapsed `00:00:32`, on `dgxh-3`, from
`2026-06-24T15:08:55` to `2026-06-24T15:09:27`.

Root cause: the prepared env had been built at
`/nfs/hpc/share/peterj29/pg/envs/goal3-cu128.tmp.20487397` and then moved to
`/nfs/hpc/share/peterj29/pg/envs/goal3-cu128`. The moved env worked through
`bin/python`, but `bin/activate` and console scripts still pointed at the old
tmp path, causing the campaign runtime check to use `/usr/bin/python` and report
missing `torch`, `triton`, `sentencepiece`, and `brotli`.

Fixes applied locally and copied to
`/nfs/hpc/share/peterj29/pg/src/pg` on 2026-06-29:

- `goal3_activate_env` now uses the final env path directly and verifies
  `python`/`sys.prefix` instead of sourcing relocatable metadata.
- `run_candidate.sh` now launches with `python -m torch.distributed.run`
  instead of the `torchrun` console script.
- `prepare-env.sbatch` repairs venv metadata after moving a tmp env into place.
- `h100-campaign-runner.sbatch` now runs a bounded `codex exec` repair pass on
  nonzero exit by default (`GOAL3_AUTO_REPAIR_ON_FAILURE=1`), without nested
  Slurm submission.

Remote verification passed: Bash syntax checks, `static_goal3_audit.py`,
Python 3.12 env imports for `torch`, `triton`, `sentencepiece`, `brotli`, and
`flash_attn_interface`, `python -m torch.distributed.run --help`, zero stale
`goal3-cu128.tmp.20487397` references, and `codex-cli 0.130.0`.

Latest fixed `srun --test-only` for the exact `dgxh`, `h100&vram80g`, `gpu:8`,
`cpus-per-task=64`, `mem=500G`, `time=06:00:00` request returned dry-run job
`20516291`, predicting start at `2026-07-02T09:17:55` on `dgxh-3`. The actual
job started earlier, at `2026-07-01T07:32:15`.

Live storage had enough headroom at the pre-submit check: `1.4T` available on
`/nfs/hpc/share/peterj29`, with the two required CaseOps data exports about
`3.2G` total. Runtime temporary files and PyTorch/Triton/PIP/CUDA caches are
routed to `/scratch/$USER/$SLURM_JOB_ID/goal3`.

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
HEAD: 705af2e4003e6b66cd050d3f0e56adbaa5d5c287
dirty: yes
```

Observed local dirty state:

```text
M goal-3/0-loop.md
M goal-3/0-plan.md
M goal-3/0-prompt.md
M goal-3/6-runner.md
M goal-3/7-approval.md
M goal-3/8-campaign.md
M goal-3/findings-summary.md
M goal-3/h100-campaign-runner.sbatch
M goal-3/h100-short-smoke.sbatch
M goal-3/scripts/common.sh
M goal-3/scripts/static_goal3_audit.py
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

Remote checkout snapshot from 2026-06-23 18:26:

```text
branch: mac
HEAD: e45bef8afd9c4129850aae14f4d0c1fd8543fbad
goal-3 status: ?? goal-3/
```

Important note: the remote Git `HEAD` is older than local, and `goal-3/` is
untracked in the remote checkout. The Goal 3 runner preserves a
`source-snapshot/goal-3/` plus `source-snapshot.sha256` inside each run
directory so the actual scripts used by a campaign are captured even when Git
`HEAD` alone is not enough to reconstruct the run.

Remote Goal 3 sync from 2026-06-23 16:41:

```text
goal-3/ synced to /nfs/hpc/share/peterj29/pg/src/pg/goal-3
remote static checks: passed
remote git status for Goal 3: ?? goal-3/
remote parameter-golf status: m parameter-golf
```

Remote Goal 3 sync from 2026-06-23 17:41:

```text
goal-3/ synced to /nfs/hpc/share/peterj29/pg/src/pg/goal-3
remote static checks: passed
remote static audit: static_goal3_audit: passed
remote log dir check: goal-3/logs/.gitkeep present
```

Remote Goal 3 sync from 2026-06-23 18:23:

```text
goal-3/ synced to /nfs/hpc/share/peterj29/pg/src/pg/goal-3
remote static checks: passed
remote static audit: static_goal3_audit: passed
remote log dir check: goal-3/logs/.gitkeep present
```

Remote static checks run in the latest sync:

```text
bash -n goal-3/scripts/common.sh
bash -n goal-3/scripts/run_candidate.sh
bash -n goal-3/prepare-env.sbatch
bash -n goal-3/prepare-tools.sbatch
bash -n goal-3/h100-env-smoke.sbatch
bash -n goal-3/h100-short-smoke.sbatch
bash -n goal-3/h100-record-runner.sbatch
bash -n goal-3/h100-campaign-runner.sbatch
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
time: 2026-06-23T18:26:36-07:00
user: peterj29
association: coehpc|eecs|peterj29|||normal
current queue: no jobs listed for peterj29
```

Live storage check from 2026-06-23 19:24:

```text
/nfs/hpc/share/peterj29: 1.5T total, 165G used, 1.4T available
/nfs/hpc/share/peterj29/pg/goal-3-runs: 996K
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched: 1.6G
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384: 1.6G
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
dgxh-1: gpu:h100-40g:16, mixed-, features h100,vram40g
dgxh-2: gpu:8, drained*, features h100,vram80g
dgxh-3: gpu:8, mixed-, features h100,vram80g
dgxh-4: gpu:8, mixed-, features h200,vram80g,vram140g
```

Dry-run findings:

```text
--constraint=h100:
  previously predicted dgxh-1
  not acceptable for the intended 8xH100 80GB run because dgxh-1 is h100-40g

old goal-3/h100-env-smoke.sbatch equivalent:
  predicted dgxh-3 at 2026-06-27T20:29:30
  stale for approval because the target is no longer a 15-minute env smoke

old goal-3/h100-record-runner.sbatch equivalent:
  predicted dgxh-3 at 2026-06-27T20:29:30
  stale for approval because the target is no longer a one-hour runner

required goal-3/h100-campaign-runner.sbatch equivalent:
  srun --test-only -p dgxh --constraint="h100&vram80g" --gres=gpu:8
  --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=500G --time=06:00:00 true
  Job 20487754 to start at 2026-06-27T20:29:30 using 64 processors on dgxh-3

rejected longer-padding check, run only to confirm whether more than six hours
was schedulable and not submitted as a job:
  srun --test-only ... --time=08:00:00 true
  rejected with MaxGRESRunMinsPerUser / accounting-QOS policy
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

Ask the user to approve or reject this exact H100 request:

```text
script: goal-3/h100-campaign-runner.sbatch
resources: dgxh, h100&vram80g, gpu:8, nodes=1, ntasks=1,
           cpus-per-task=64, mem=500G, time=06:00:00
dry-run: Job 20487754 predicted start 2026-06-27T20:29:30 on dgxh-3
sequence: runtime validation -> short dense/qMLP/TTT smoke gate ->
          dense_sp8192 seed 42 baseline hard gate ->
          qmlp_sp16384 seeds 42, 0, 1234
preserved source: source-snapshot/goal-3/ and source-snapshot.sha256
```

Do not submit the campaign, env smoke, short smoke, or record runner until that
exact corresponding request is explicitly approved.

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

Runtime qMLP and qMLP+TTT smokes are still pending inside the approved H100
campaign.

## Script Artifacts

Created and statically checked:

```text
goal-3/prepare-env.sbatch
goal-3/h100-env-smoke.sbatch
goal-3/h100-short-smoke.sbatch
goal-3/h100-record-runner.sbatch
goal-3/h100-campaign-runner.sbatch
goal-3/h100-repair-agent.sbatch
goal-3/scripts/common.sh
goal-3/scripts/env_smoke.py
goal-3/scripts/run_candidate.sh
goal-3/scripts/parse_train_log.py
```

Default campaign runner order:

```text
runtime validation -> short dense/qMLP/TTT smoke gate ->
dense_sp8192 seed 42 baseline hard gate ->
qmlp_sp16384 seeds 42, 0, 1234
```

H100 approval status: campaign request prepared in `goal-3/7-approval.md`;
not yet granted. The exact six-hour dry-run is recorded.

Additional prep now complete:

- `goal-3/compliance-note.md` states the qMLP compliance assumptions and the
  runtime proof still required.
- CPU environment prep completed as job `20487397`, producing
  `/nfs/hpc/share/peterj29/pg/envs/goal3-cu128`.
- CPU tools prep completed as job `20487617`, producing
  `/nfs/hpc/share/peterj29/pg/tools/lrzip/bin/lrzip` with user-local LZO and
  LZ4 libraries. Direct submit-node execution of that binary fails due the
  submit node's older glibc, so the valid runtime check is the campaign-internal
  H100 env smoke.
- `goal-3/scripts/env_smoke.py` checks CUDA device count, FA3 import, `lrzip`
  presence and executability, and tokenizer vocab sizes.
- `goal-3/scripts/parse_train_log.py` now reports both the last periodic
  train-loss step and the final wallclock stop step as `train_steps_final`.
- `goal-3/scripts/static_goal3_audit.py` checks qMLP flag/component wiring,
  qMLP Hessian and compression key coverage, candidate-to-vocab mappings,
  H100 80GB constraints, final-status traps, qMLP+TTT smoke coverage, and
  bounded runner defaults.
  It intentionally avoids newer Python-only syntax because the submit-node
  `python3` used for static checks is older than the local Mac Python.
- `goal-3/scripts/run_candidate.sh` launches `torchrun` through `srun` by
  default inside Slurm allocations and records the effective launcher in each
  candidate `env.txt`.
- `goal-3/scripts/common.sh` now prepares runtime storage before Python or FA3
  setup: `TMPDIR`, `PIP_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
  `TRITON_CACHE_DIR`, and `CUDA_CACHE_PATH` are routed to
  `/scratch/$USER/$SLURM_JOB_ID/goal3`, and `runtime-storage.txt` records the
  effective paths.
- `goal-3/h100-record-runner.sbatch` now bounds the default one-hour candidate
  timeouts to `8m + 8m + 36m`; `goal-3/h100-short-smoke.sbatch` now defaults
  to `10m` per smoke candidate inside its 45-minute allocation.
- `goal-3/h100-campaign-runner.sbatch` is now the primary H100 approval target,
  with a six-hour request, runtime setup validation, short dense/qMLP/TTT
  candidate smoke gate, a full dense baseline hard validity gate plus strict
  parity note, and qMLP `sp16384` seeds `42`, `0`, and `1234`.
- `goal-3/h100-short-smoke.sbatch` now uses the same four-smoke default as the
  campaign runner: `dense_sp8192_smoke`, `qmlp_sp8192_smoke`,
  `qmlp_sp16384_smoke`, and `qmlp_sp16384_ttt_smoke`.
- Goal 3 runs now preserve `source-snapshot/goal-3/` and
  `source-snapshot.sha256` in each run directory, so untracked or dirty Goal 3
  files are captured even if Git `HEAD` alone is not reproducible.
- `goal-3/logs/.gitkeep` now preserves the sbatch stdout/stderr directory; the
  static audit fails if this placeholder is missing.
- `run_candidate.sh` now derives `RUN_ID` from `${candidate}_seed${seed}` by
  default, preventing accidental reuse of TTT `/tmp` counter paths across
  sequential campaign candidates.
- `h100-campaign-runner.sbatch` now writes a campaign-aware `final-status.json`
  on early exits, including any available env-smoke, smoke-gate,
  baseline-parity, candidate summary/status, artifact manifest, and
  source-snapshot evidence.
- Campaign smoke, baseline, and final summary gate writers now tolerate missing
  or malformed candidate JSON by recording structured `_load_error` diagnostic
  payloads instead of failing inside the gate writer.
- Normal successful campaign `final-status.json` now includes the same context,
  environment-smoke, smoke-gate, baseline, source-snapshot, and file pointer
  fields expected from early-exit summaries.
- Campaign allocation is now `06:00:00` and full-candidate timeout is now
  `120m`, because we do not want qMLP to fail due arbitrary padding constraints
  around quantization, compression, TTT compile warmup, or TTT eval.
- Baseline handling now has two tiers: strict parity target `BPB<=1.065` and
  `steps>=4500` for interpretation, and hard stop gate `BPB<=1.075` and
  `steps>=4000` for campaign control. A borderline baseline should caveat the
  qMLP comparison, not automatically end the allocation.
- Local static checks passed on 2026-06-23 at 18:23 Pacific after the
  qMLP+TTT smoke/default update.
- Remote submit-node static checks passed on 2026-06-23 at 18:23 Pacific after
  syncing the same update through the OSU gateway.
- Candidate `status.json` now records `timeout` and `timed_out` so a timeout
  kill is explicit in machine-readable summaries.
- H100 env, short-smoke, record-runner, campaign-runner, and optional
  repair-agent scripts now use `goal3_write_final_status` traps so early
  failures still leave `final-status.json` when possible.
- H100 short-smoke and record-runner normal summaries include candidate order,
  seed, timeout settings, and per-candidate `status.json` contents.
- `goal-3/scripts/common.sh` exports required Goal 3 paths for child processes.
- `goal-3/scripts/common.sh` records `git-status.txt`, `git-diff.stat`, and
  `git-diff.patch`.
- `goal-3/h100-short-smoke.sbatch`, `goal-3/h100-record-runner.sbatch`, and
  `goal-3/h100-campaign-runner.sbatch` stage the Goal 3 source tree and both
  required CaseOps data/tokenizer sets to `/scratch/$USER/$SLURM_JOB_ID/goal3`.
