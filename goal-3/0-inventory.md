# Phase 0: Inventory

## Overview

Establish the Goal 3 source of truth before implementation or H100 submission.
This phase reads the governing docs, records local and remote state, refreshes
live OSU/HPC facts, and identifies the immediate next phase.

## Implementation Steps

1. Read Goal 3 governing docs:
   - `goal-3/0-plan.md`;
   - `goal-3/0-loop.md`;
   - `goal-3/0-prompt.md`.
2. Read carry-forward evidence from `goal-2/findings-summary.md`.
3. Read OSU/HPC operating docs:
   - `osu-hpc-agent-guide/AGENTS.md`;
   - `osu-hpc-agent-guide/ssh-setup.md`;
   - relevant sections of `osu-hpc-agent-guide/OSU_HPC_AGENT_GUIDE.md`.
4. Read the primary and fallback record READMEs.
5. Inspect the primary base `train_gpt.py` structure around:
   - Hyperparameters;
   - MLP;
   - GPT forward and `forward_ttt`;
   - TTT LoRA modules;
   - optimizer grouping;
   - GPTQ/LQER;
   - per-group compression and serialization.
6. Record local Git and submodule state.
7. Refresh live HPC state through the documented OSU path without submitting
   jobs.
8. Create `goal-3/status.md`, `goal-3/jobs.csv`, and
   `goal-3/findings-summary.md`.

## Verification

Completed read-only checks:

```text
local date/git/submodule state
remote submit host/date/codex check
squeue -u peterj29
sacctmgr association and QOS check
df for home and hpc-share
sinfo dgxh node check
srun --test-only for h100-only and h100&vram80g constraints
remote project checkout location and git state
```

No training, data export, GPU diagnostic, compilation, or H100/H200 job was run.

## Findings

- Local branch is `mac` at
  `aa7b89cee2c29bc1a0acbe7d57f100aa27c58ed9`.
- Local worktree is dirty because `parameter-golf` is modified and `goal-3/`
  has new docs.
- Remote checkout exists at `/nfs/hpc/share/peterj29/pg/src/pg`, branch `mac`,
  HEAD `e45bef8afd9c4129850aae14f4d0c1fd8543fbad`, and is stale relative to
  local Goal 3 docs.
- Live Slurm shows no queued/running jobs for `peterj29`.
- `codex` is installed on the HPC submit node at
  `/nfs/stak/users/peterj29/.local/bin/codex`, version `0.130.0`.
- `dgxh` QOS currently shows `gres/gpu=8` per-user and `gres/gpu=2880`
  run-minutes per user.
- `--constraint=h100` alone can target `dgxh-1`, which is `h100-40g`, so it is
  not specific enough for the intended competition-class node.
- `--constraint="h100&vram80g"` dry-runs to `dgxh-3`, the intended H100 80GB
  class.

## Completion Requirements

- No H100/H200 jobs submitted: complete.
- Live scheduler/account facts recorded with timestamp: complete.
- Selected base and fallback recorded: complete.
- Current next action clear in `goal-3/status.md`: complete.

## Next Phase

Phase 1: write `goal-3/1-base.md` with the base stack comparison and
compliance-sensitive code map.
