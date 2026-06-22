# OSU HPC instructions for coding agents

These instructions apply whenever this repository is used on the Oregon State University College of Engineering HPC cluster. Read `OSU_HPC_AGENT_GUIDE.md` for the complete reference.

## Authority

Use this order when information conflicts:

1. Current OSU policy or administrator direction.
2. Live Slurm, module, quota, mount, and node information.
3. Current OSU HPC web documentation.
4. This file and the companion guide.

The cluster changes frequently. Never hard-code a node count, GPU count, GRES name, partition limit, module version, or quota without checking it live.

## Safety boundary

- Work only as the authenticated user. Never use `sudo`, attempt privilege escalation, change system configuration, or access another user's files or jobs.
- Never run network scans, cryptocurrency mining, unrelated distributed computing, public listeners, file-sharing services, web servers, or persistent daemons.
- Jupyter or other interactive services must use OSU Open OnDemand or an allocated compute node with an approved localhost tunnel.
- Never expose secrets in Git, command arguments, `#SBATCH` directives, job names, logs, filenames, diagnostics, or shell tracing.
- Never cancel all jobs, delete shared data, overwrite an environment in active use, or perform broad recursive deletion without explicit human approval.
- Do not alter `/apps`, global modules, drivers, Slurm configuration, system services, or licensed-software configuration.
- Do not make account, QOS, partition, license, software-installation, or data-governance requests on the user's behalf; prepare the request for human review.

## Submit-node rule

Submit nodes are control planes, not compute nodes.

Allowed on submit nodes:

- edit and inspect source;
- run Git commands;
- inspect modules, Slurm, quotas, and small text files;
- create batch scripts;
- submit, inspect, and cancel specifically identified jobs;
- perform lightweight static checks.

Use a Slurm allocation for:

- compilation of a material project;
- unit/integration tests that use sustained CPU or significant RAM;
- dataset preprocessing;
- solvers, simulations, model inference or training;
- GPU commands;
- profilers, notebook kernels, databases, or containers that perform real work;
- operations that open or create many files;
- any long-running command.

If uncertain, use a small allocation.

## First actions in a new session

Run read-only checks:

```bash
date
hostname -f
module list
sinfo
squeue -u "$USER"
bash bin/osu-hpc-inventory.sh
```

Review the generated report before choosing resources. Do not fan out with SSH across compute nodes.

## Access model

- Connect through `submit-a`, `submit-b`, or `submit-c`.
- Direct compute-node SSH is valid only while the user owns an allocation there.
- Prefer `sbatch` for unattended or long jobs. An interactive `srun` can die when the submit-side session terminates.
- Off-campus access normally requires OSU VPN or the Engineering gateway.
- The browser portal requires campus/VPN access.

## Slurm requirements

Every production script should explicitly set:

```text
job name
account, when needed
partition
walltime
nodes
tasks
CPUs per task
memory
GPU count, when needed
constraints, when needed
stdout/stderr paths
```

Do not rely on default walltime, memory, partition, or module versions.

A feature does not allocate a GPU. A GPU job needs both a resource request and, when appropriate, a model constraint:

```bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
```

Do not request an exact node unless there is a documented reproducibility need.

Use:

- `--cpus-per-task=N` for threads used by one process;
- `--ntasks=N` for independent or MPI processes;
- `--mem=X` for memory per node;
- `--mem-per-cpu=X` only when memory should scale with CPUs;
- job arrays with a concurrency cap for independent parameter sweeps.

Start with the smallest representative allocation. Scale only after inspecting runtime, CPU efficiency, RSS, GPU memory/utilization, and I/O.

## Resource ceilings

Public documentation lists shared access to `share`, `dgx2`, `dgxh`, and `preempt`, but access and QOS are account-specific. Query the live scheduler. Public limits may have changed.

Never exceed a repository-specific ceiling stated below. In the absence of one, require human approval before requesting any of:

- more than 1 node;
- more than 1 GPU;
- more than 16 CPUs;
- more than 128 GB RAM;
- more than 8 hours;
- more than 20 simultaneous array elements.

This is an agent approval boundary, not a claim about OSU's scheduler limit.

## GPU protocol

Before depending on a GPU class:

```bash
sinfo -N -o '%N|%P|%T|%G|%f'
scontrol show partition PARTITION
```

Inside an allocated GPU job:

```bash
nvidia-smi -L
nvidia-smi
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-}"
module list
nvcc --version 2>/dev/null || true
```

The public cluster inventory includes H200, H100, L40S, A40, V100/V100S, RTX 8000, RTX 6000, T4, M60, and some private-partition GPUs. MIG configuration is mutable. Use live GRES/features.

Do not reserve a GPU during long CPU-only preparation. End idle interactive GPU allocations promptly.

## Environment protocol

Begin batch jobs with a clean, versioned environment:

```bash
module purge
command -v srun >/dev/null 2>&1 || module load slurm
module load PACKAGE/VERSION
module list
```

Use the same compiler/MPI modules at build and run time.

Store Python/Conda environments and Apptainer images under `/nfs/hpc/share/$USER`, not the 25 GB home directory. Do not `pip install` into the system interpreter or mutate an environment while jobs use it. Prefer locked, versioned environments.

For containers, use Apptainer, pin the image digest, preserve the definition file/checksum, and use `--nv` for GPU access when appropriate.

## Storage protocol

- `$HOME`: small configuration and source only; documented fixed quota is 25 GB.
- `/nfs/hpc/share/$USER`: shared scratch/workspace; not backed up, no file recovery, and potentially subject to purge. Live quota is authoritative.
- `/scratch/$USER/$SLURM_JOB_ID`: node-local high-performance temporary data; not backed up and may disappear without warning.
- Project/archive storage: durable copy for important data.

For heavy I/O, stage inputs to local `/scratch`, run there, then copy validated outputs to shared/durable storage. Use a job-specific path and a guarded cleanup trap.

Never delete the only copy of any input, checkpoint, result, or environment. Before `rm -rf` or `rsync --delete`, resolve and validate the path and require human approval for a broad operation.

Avoid continuous `du`, recursive `find`, tight filesystem polling, millions of tiny files, or excessive DataLoader workers on `hpc-share`.

## Batch-script quality

Use:

```bash
set -euo pipefail
```

Create unique run and log paths using the Slurm job ID. Record at least:

```bash
date --iso-8601=seconds
hostname -f
pwd
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module -t list 2>&1 || true
env | grep '^SLURM_' | sort
```

For GPU jobs, also record `nvidia-smi`. For Python, record `python -VV` and a dependency snapshot. Never dump the full environment because it may contain secrets.

Write outputs atomically: write to a temporary file, validate it, then rename. Do not allow multiple jobs to write the same filename.

Use `srun` for workload steps inside an allocation unless the application's documented launcher requires otherwise.

## Checkpoint and preemption rules

Treat `preempt` termination as normal. Use it only for restartable work.

A long or preemptible job must have:

- a tested checkpoint mechanism;
- idempotent resume behavior;
- a signal handler or periodic checkpoint;
- stage-out that can safely run more than once;
- a manifest showing the latest validated checkpoint.

Do not submit an uncheckpointed multi-day run.

## Monitoring

Capture the job ID:

```bash
job_id=$(sbatch --parsable job.sbatch)
```

Poll no faster than necessary—normally 30–60 seconds for a short smoke test and several minutes for a long job. Stop after a terminal state. Do not continuously monitor shared filesystems.

Use:

```bash
squeue -h -j "$job_id" -o '%T'
scontrol show job "$job_id"
sacct -j "$job_id" -X \
  -o JobID,State,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS,ExitCode
```

A job leaving `squeue` is not proof of success.

## Completion definition

Do not report success until all applicable conditions hold:

- Slurm state is `COMPLETED`;
- exit code is successful;
- expected outputs exist and are nonempty;
- project tests/validation pass;
- stage-out completed;
- important results are no longer local-scratch-only;
- run manifest, modules, code revision, configuration, and logs are preserved;
- resource use is reviewed before scaling.

Explicitly report `FAILED`, `OUT_OF_MEMORY`, `TIMEOUT`, `PREEMPTED`, `CANCELLED`, or `NODE_FAIL`.

## Failure handling

- Pending: inspect `%R` and `scontrol show job`; do not repeatedly cancel/resubmit without cause.
- OOM: inspect `MaxRSS`; reduce working set or request measured memory plus margin.
- Timeout: checkpoint/split/optimize or request a realistic permitted walltime.
- GPU invisible: verify allocation GRES, `CUDA_VISIBLE_DEVICES`, container `--nv`, and framework build.
- Illegal instruction: rebuild conservatively or constrain to the CPU feature used at build time.
- Quota: identify regenerable data; do not delete broadly.
- Node failure: preserve evidence and resume from a checkpoint.
- Access/configuration problem: prepare a support ticket with `COE HPC` in the subject, job ID, exact command/error, account, partition, host, and timestamp. Do not include secrets.

## Change discipline

Before editing:

```bash
git status --short
```

Do not discard unrelated human changes. Keep scheduler/environment changes separate from scientific-code changes where practical.

Before submission:

```bash
bash -n job.sbatch
grep '^#SBATCH' job.sbatch
git diff --check
```

Run the smallest relevant test inside an allocation. Show the human the resource request and any destructive or high-cost action before execution.

## Repository-specific instructions

Fill these in for the project:

```text
Project root:
Build command:
Unit-test command:
Integration-test command:
Environment/modules:
Default partition/account:
Maximum agent-approved resources:
Input-data location:
Durable output location:
Local-scratch staging procedure:
Checkpoint command/interval:
Expected outputs:
Validation/acceptance criteria:
Sensitive-data restrictions:
Licensed-software requirements:
```
