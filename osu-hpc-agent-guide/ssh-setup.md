# SSH Setup for OSU HPC

This note records the local shortcut workflow for reaching the OSU Engineering
gateway and then the COE HPC submit node.

## Observed Shortcuts

On the local Mac, `osu` is defined in `~/.zshrc` as:

```bash
alias osu='ssh peterj29@access.engr.oregonstate.edu'
```

After logging into that OSU Engineering gateway, `hpc` is defined there as:

```bash
alias hpc='ssh peterj29@submit-a.hpc.engr.oregonstate.edu'
```

So the intended interactive path is:

```bash
osu
hpc
```

The first command reaches the Engineering access/flip gateway. The second
command, run from that gateway shell, reaches the HPC submit node
`submit-a.hpc.engr.oregonstate.edu`.

## Host Roles

- `access.engr.oregonstate.edu` is the Engineering gateway/flip host used to
  reach internal Engineering resources from off campus or without VPN.
- `submit-a.hpc.engr.oregonstate.edu` is an HPC submit/control node. Use it for
  editing, Git, light inspection, Slurm commands, and job submission.
- Compute nodes are not reached by the `hpc` alias. Request compute resources
  from the submit node through Slurm with `sbatch` or `srun`.

Do not run training, dataset preprocessing, benchmarks, GPU commands, or other
heavy work on either the gateway or the submit node.

## Common Interactive Workflow

From the local Mac:

```bash
osu
```

From the OSU gateway prompt:

```bash
hpc
```

From the HPC submit node:

```bash
hostname -f
date --iso-8601=seconds
module list
sinfo
squeue -u "$USER"
```

For an interactive compute allocation from the submit node:

```bash
srun \
  --partition=share \
  --time=01:00:00 \
  --cpus-per-task=4 \
  --mem=16G \
  --pty bash
```

For an unattended job, prefer `sbatch` from the submit node:

```bash
job_id=$(sbatch --parsable job.sbatch)
printf 'submitted %s\n' "$job_id"
```

## Agent/Automation Notes

The `osu` shortcut is local, while `hpc` is available after entering the OSU
gateway shell. Interactive use should prefer the two explicit commands:

```bash
osu
hpc
```

For noninteractive automation, shell aliases can be brittle because they only
expand in the shell where they are defined. The direct equivalent is:

```bash
ssh peterj29@access.engr.oregonstate.edu \
  "ssh peterj29@submit-a.hpc.engr.oregonstate.edu 'hostname -f; squeue -u \$USER'"
```

Use noninteractive SSH only for lightweight inspection, Slurm control, and file
operations. Material compute should still be submitted to Slurm and run inside
an allocation.

## Quick Verification

Local check:

```bash
type osu
```

Gateway check:

```bash
osu
type hpc
```

Submit-node check:

```bash
hpc
hostname -f
scontrol --version
```

If `hpc` is not found on the gateway, use the direct submit-node command:

```bash
ssh peterj29@submit-a.hpc.engr.oregonstate.edu
```

If direct submit-node SSH fails from the local Mac, enter through `osu` first.
