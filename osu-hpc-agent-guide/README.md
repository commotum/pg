# Agent's Guide to OSU College of Engineering HPC

This package is a practical operating guide for humans and coding agents using the Oregon State University College of Engineering HPC cluster.

## Contents

- `OSU_HPC_AGENT_GUIDE.md` — comprehensive architecture, access, Slurm, storage, software, policy, GPU, troubleshooting, and agent-operation reference.
- `ssh-setup.md` — local `osu` and gateway-side `hpc` SSH shortcut workflow.
- `AGENTS.md` — concise Codex instructions suitable for a repository root or `~/.codex/AGENTS.md`.
- `LIVE_INVENTORY_CHECKLIST.md` — commands and interpretation notes for reconciling the public documentation with the live cluster.
- `data/public-baseline.json` — machine-readable public-document baseline for agent tooling; never treat it as live state.
- `bin/osu-hpc-inventory.sh` — read-only live inventory collector. Run it on an OSU HPC submit node.
- `bin/osu-hpc-job-report.sh` — summarize a completed or running Slurm job.
- `templates/` — conservative CPU, GPU, array, MPI, and preemptible Slurm templates.

## Recommended first use

```bash
scp -r osu-hpc-agent-guide \
  "$ONID@submit.hpc.engr.oregonstate.edu:~/"

ssh "$ONID@submit.hpc.engr.oregonstate.edu"
cd ~/osu-hpc-agent-guide
bash bin/osu-hpc-inventory.sh
```

The inventory report is written beneath `inventory/`. Review it before selecting a partition, GPU constraint, module version, or quota.

## Authority rule

The public documentation is a planning reference, not a live scheduler database. For current operation, use this order of authority:

1. Current OSU policy and administrator instructions.
2. Live Slurm configuration and node state.
3. Live filesystem quota/mount information and module inventory.
4. The public OSU HPC documentation.
5. This guide's recommendations.

Public pages retrieved for this edition were reviewed on **2026-06-20**. The public status page's latest visible update at review time was **2026-06-18 at 6 p.m.**, after a cluster-wide maintenance window, and said that EL9 GPU and DGX nodes were still being restored. Always run the live inventory.

## Intended use

Copy `AGENTS.md` into a repository that will be operated on the cluster, then add project-specific build, test, input-data, checkpoint, and acceptance instructions below it. Keep the comprehensive guide outside Codex's default instruction budget and link to it from the repository instructions.
