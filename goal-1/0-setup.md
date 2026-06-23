# Phase 0: Remote Project Setup

Date drafted: 2026-06-21

## Overview

Create or refresh the remote OSU HPC working copy under shared HPC storage, confirm the expected branch and submodules are available there, and capture a live inventory snapshot from the submit node.

This phase does not run training, dataset preparation, GPU commands, compilation, or any other compute-heavy work. It only prepares the control-plane state needed for later Slurm jobs.

## Why This Matters

The qMLP question cannot be answered from the Mac alone because the meaningful benchmarks need CUDA GPUs. Before spending any GPU time, we need a reproducible remote project location, checked-out dependencies, and current scheduler/storage facts.

This phase moves the project forward by making later environment smoke tests and benchmarks executable from the HPC side.

## Assumptions and Dependencies

- Local branch is `mac`.
- Remote SSH path is Mac -> OSU flip -> HPC submit.
- Local alias `osu` reaches `peterj29@access.engr.oregonstate.edu`.
- Flip-side alias `hpc` reaches an HPC submit node.
- Remote shared storage path is `/nfs/hpc/share/$USER`.
- The repo can be cloned or updated from GitHub on the HPC submit node.
- The `osu-hpc-agent-guide` folder is available in the repo.
- Inventory scripts are safe to run on the submit node because they inspect scheduler and environment state rather than doing compute work.

## Implementation Steps

1. Confirm current local branch and submodule state.

```bash
git branch --show-current
git submodule status
```

2. On the HPC submit node, create the project source directory.

```bash
mkdir -p /nfs/hpc/share/$USER/pg/src
```

3. Clone the repo if it does not exist remotely; otherwise fetch and update it.

```bash
cd /nfs/hpc/share/$USER/pg/src
if [ ! -d pg/.git ]; then
  git clone git@github.com:commotum/pg.git pg
fi
cd pg
git fetch origin
git checkout mac
git pull --ff-only
git submodule update --init --recursive
```

4. Capture remote repo state.

```bash
pwd
git branch --show-current
git rev-parse HEAD
git status --short
git submodule status
```

5. Run the HPC inventory script from the guide.

```bash
cd /nfs/hpc/share/$USER/pg/src/pg/osu-hpc-agent-guide
bash bin/osu-hpc-inventory.sh
```

6. Preserve a concise inventory record under the run tree.

Use a path like:

```text
/nfs/hpc/share/$USER/pg/runs/inventory/YYYYMMDD-HHMMSS/
```

The record should include:

- hostname;
- date;
- repo path and commit;
- branch;
- submodule status;
- `sinfo` summary;
- current user queue;
- relevant storage paths and free space.

## Expected Artifacts

- Remote repo at `/nfs/hpc/share/$USER/pg/src/pg`.
- Remote inventory directory under `/nfs/hpc/share/$USER/pg/runs/inventory/`.
- This phase file updated with the actual result.
- Any stale facts in `goal/0-plan.md` corrected or timestamped if the live inventory differs.

## Completion Requirements

This phase is complete when:

- The remote repo exists under `/nfs/hpc/share/$USER/pg/src/pg`.
- The remote repo is on branch `mac` or a documented successor branch.
- The remote repo has initialized `parameter-golf` and `qham` submodules.
- A live inventory snapshot is saved under `/nfs/hpc/share/$USER/pg/runs/inventory/`.
- No training, dataset preparation, GPU checks, compilation, or compute-heavy work has run on the submit node.
- The result section below records evidence, artifacts, new facts, and the decision for Phase 1.

## Failure and Fallback Rules

- If GitHub SSH clone fails, test whether HTTPS clone is viable before declaring the phase blocked.
- If the remote repo has local uncommitted changes, do not overwrite them; capture `git status --short` and decide whether to use a separate fresh worktree.
- If `git pull --ff-only` fails, do not merge on the remote; capture the error and choose a clean branch or fresh clone.
- If the inventory script fails because of a missing command, collect the equivalent facts manually with simple scheduler and filesystem commands.
- If SSH access fails, record the exact failed hop and stop before changing local SSH configuration.

## Result

Status: complete

Evidence:

- Remote source directory exists at `/nfs/hpc/share/peterj29/pg/src/pg`.
- Remote host was `submit-a.ib.coehpc`.
- Remote branch is `mac`.
- Remote commit is `e45bef8afd9c4129850aae14f4d0c1fd8543fbad`.
- Remote `git status --short` is clean after removing the generated source-tree inventory copy.
- Remote submodules are initialized:
  - `parameter-golf` at `f5c079314c4877fbb0af378c0abade5a8ca33d3a`.
  - `qham` at `fb7b546294aecdabace2f5fab0527001df320b77`.
- Inventory capture timestamp from the guide: `2026-06-21T22:26:40-07:00`.
- Submit-node GPU runtime inspection intentionally skipped because the inventory script does not request GPUs.
- No training, data preparation, compilation, allocation, or GPU probe was run on the submit node.

Artifacts:

- `/nfs/hpc/share/peterj29/pg/runs/inventory/20260621-222534/summary.txt`
- `/nfs/hpc/share/peterj29/pg/runs/inventory/20260621-222534/inventory.txt`
- `/nfs/hpc/share/peterj29/pg/runs/inventory/20260621-222534/guide-inventory/`

New facts:

- The remote repo did not exist before this phase; it was cloned fresh.
- The corrected nested SSH form is:

```bash
ssh peterj29@access.engr.oregonstate.edu "ssh submit-a.hpc.engr.oregonstate.edu '...'"
```

- Without the corrected quoting, chained commands after the nested SSH can accidentally run on the flip host.
- The current visible Slurm association remains `coehpc|eecs|peterj29|||normal||||`.
- The current user queue was empty at inventory time.
- Current loaded modules on submit are only `slurm/current`.
- Storage at inventory time:
  - `$HOME` maps to `/nfs/stak/users/peterj29`, 25G total, 12G used, 14G available.
  - `/nfs/hpc/share/peterj29` is Lustre, 1.5T total, 23G used, about 1.5T available.
  - `/scratch` exists on submit, 347G total, 344G available, but compute jobs should use compute-node scratch inside allocations.
- The generated guide inventory initially dirtied the remote source tree under `osu-hpc-agent-guide/inventory/`; it was copied to the run artifact directory and then removed from the source tree.

Decision:

- Continue to Phase 1: Environment Smoke Test.
- Phase 1 should create `goal/1-smoke.md` before implementation.
- The first GPU allocation should remain a short Slurm smoke job on `gpu` with `rtx8000`, unless a fresh scheduler test shows that target is no longer suitable.
