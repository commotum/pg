# Phase 1: Environment Smoke Test

Date drafted: 2026-06-21

## Overview

Create a minimal shared Python environment on OSU HPC and run a short one-GPU Slurm job that proves CUDA hardware visibility and PyTorch GPU execution.

This phase is not a training benchmark. It should not download datasets, run Parameter Golf training, compile custom kernels, or use scarce GPUs.

## Why This Matters

The qMLP experiments need a CUDA-capable Python environment before any baseline or quaternion benchmark can be trusted. A tiny GPU smoke test catches the common failures early: wrong Python, missing CUDA runtime, CPU-only PyTorch, broken Slurm GPU allocation, or job artifact paths that do not survive after the allocation ends.

This phase moves us closer to answering the qMLP question by proving the execution substrate for later controlled experiments.

## Assumptions and Dependencies

- Phase 0 is complete.
- Remote repo path is `/nfs/hpc/share/peterj29/pg/src/pg`.
- Run root is `/nfs/hpc/share/peterj29/pg/runs`.
- Shared environments should live under `/nfs/hpc/share/peterj29/pg/envs`.
- Default submit-node Python is too old for this project; use `module load python/3.11`.
- Verified module result: `python/3.11` provides Python `3.11.14` at `/usr/local/apps/python/3.11/bin/python`.
- Available CUDA modules include `cuda/13.0`; the Phase 1 venv resolved to a CUDA 13.0 PyTorch build, so the smoke job should load `cuda/13.0` with `cuda/12.8` only as a fallback.
- There is no PyTorch module visible in the Phase 0 inventory, so create a versioned venv and install PyTorch into it.

## Resource Request

Use the cheap smoke-test GPU class unless a fresh scheduler test contradicts it:

```text
partition=gpu
constraint=rtx8000
gres=gpu:1
time=00:10:00
cpus-per-task=4
mem=16G
nodes=1
ntasks=1
```

This is inside the standing safety boundary: one node, one GPU, 4 CPUs, 16G RAM, and 10 minutes.

## Implementation Steps

1. Create the shared environment directory.

```bash
mkdir -p /nfs/hpc/share/peterj29/pg/envs
```

2. Create a versioned Python venv if it does not already exist.

```bash
module purge
module load python/3.11
python -m venv /nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621
```

3. Install the minimal smoke dependencies into the venv.

```bash
source /nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch numpy
python -m pip freeze --all > /nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621.freeze.txt
python -VV > /nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621.python-version.txt
```

Do not install into the system interpreter. Do not mutate this environment while jobs are using it.

4. Before submitting, run a scheduler fit check.

```bash
srun --test-only -p gpu --constraint=rtx8000 --gres=gpu:1 \
  --time=00:10:00 --cpus-per-task=4 --mem=16G true
```

5. Submit a short smoke job that records:

- hostname;
- Slurm job metadata;
- loaded modules;
- `CUDA_VISIBLE_DEVICES`;
- `nvidia-smi -L`;
- `nvidia-smi`;
- `python -VV`;
- `pip freeze --all`;
- `torch.__version__`;
- `torch.version.cuda`;
- `torch.cuda.is_available()`;
- `torch.cuda.device_count()`;
- `torch.cuda.get_device_name(0)`;
- a tiny CUDA tensor operation;
- a tiny backward pass.

6. Preserve all artifacts under:

```text
/nfs/hpc/share/peterj29/pg/runs/phase1-smoke/<job-id>/
```

## Expected Artifacts

- Shared venv at `/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621`.
- Environment freeze file beside the venv.
- Local reusable batch script at `goal/1-smoke.sbatch`.
- Slurm batch script under the Phase 1 run directory.
- Slurm stdout/stderr.
- `manifest.txt`.
- `gpu.txt`.
- `python.txt`.
- `torch-smoke.txt`.

## Completion Requirements

This phase is complete when:

- The shared venv exists under `hpc-share`.
- The venv records Python version and installed packages.
- A Slurm job with the resource request above reaches state `COMPLETED`.
- Job exit code is `0:0`.
- `nvidia-smi -L` sees exactly the allocated GPU.
- PyTorch imports from the shared venv.
- `torch.cuda.is_available()` is `True`.
- PyTorch sees exactly one CUDA device in the allocation.
- The recorded PyTorch GPU name matches the allocated GPU class closely enough to validate the allocation.
- A small CUDA tensor operation and backward pass complete.
- Artifacts are preserved under `/nfs/hpc/share/peterj29/pg/runs/phase1-smoke/<job-id>/`.

## Failure and Fallback Rules

- If PyTorch installation fails because network or package resolution is unavailable on the submit node, create a CPU Slurm job to build the environment or switch to Conda under `/nfs/hpc/share/peterj29/pg/conda-envs`.
- If `pip install torch` resolves to a CPU-only build, replace the environment with a CUDA-specific PyTorch wheel command rather than trying to patch the environment in place.
- If the RTX8000 scheduler check does not fit, rerun `srun --test-only` for `ampere/a40` and `preempt/a40` before submitting.
- If the Slurm job fails because of environment activation, fix the batch script and resubmit once.
- If the Slurm job fails because the GPU is missing, capture `scontrol show job`, job logs, and `sacct`, then choose a different GPU target.
- Do not escalate to H100/H200 for this phase.

## Result

Status: complete

Evidence:

- Shared venv exists at `/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621`.
- Submit-node import check succeeded before submission:
  - Python module: `python/3.11`, Python `3.11.14`.
  - `torch 2.12.1+cu130`.
  - `torch.version.cuda == 13.0`.
  - `torch.cuda.is_available()` was `False` on submit, as expected without an allocation.
- Scheduler fit check for `gpu/rtx8000` predicted start on `cn-gpu7`.
- Slurm job `20480372` completed:
  - state `COMPLETED`;
  - exit code `0:0`;
  - elapsed `00:00:20`;
  - allocation `cpu=4, gres/gpu=1, mem=16G, node=1`.
- Job ran on `cn-gpu7.hpc.engr.oregonstate.edu` in partition `gpu`.
- `CUDA_VISIBLE_DEVICES=0`.
- Loaded modules in the job were `slurm/current` and `cuda/13.0`.
- `nvidia-smi -L` saw exactly one GPU: `Quadro RTX 8000`.
- GPU memory was `46080 MiB`.
- Driver was `595.71.05`; `nvidia-smi` reported CUDA `13.2`.
- `nvcc` came from CUDA `13.0`, release `V13.0.88`.
- PyTorch smoke output:

```json
{
  "cuda_available": true,
  "cuda_device_count": 1,
  "device_name": "Quadro RTX 8000",
  "grad_norm": 8.98193645477295,
  "loss": 1029.932373046875,
  "max_memory_allocated": 21234688,
  "status": "ok",
  "torch_cuda": "13.0",
  "torch_version": "2.12.1+cu130"
}
```

Artifacts:

- Local batch script: `goal/1-smoke.sbatch`.
- Remote batch script: `/nfs/hpc/share/peterj29/pg/runs/phase1-smoke/phase1-smoke.sbatch`.
- Remote job directory: `/nfs/hpc/share/peterj29/pg/runs/phase1-smoke/20480372/`.
- Slurm stdout: `/nfs/hpc/share/peterj29/pg/runs/phase1-smoke/slurm-20480372.out`.
- Slurm stderr: `/nfs/hpc/share/peterj29/pg/runs/phase1-smoke/slurm-20480372.err`.
- Environment freeze: `/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621.freeze.txt`.
- Python version file: `/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621.python-version.txt`.

New facts:

- The default `pip install torch` on this cluster resolved to `torch 2.12.1+cu130` and installed CUDA 13 runtime packages from PyPI.
- CUDA 13 works on the RTX8000 node tested here.
- The smoke venv is enough for PyTorch GPU execution but is not yet a full Parameter Golf training environment; Phase 2 still needs the remaining `parameter-golf/requirements.txt` packages.
- The `gpu` partition remains useful for low-stakes smoke tests with the `eecs` account.
- Slurm output/error files for this script were empty on success because artifacts are written into the job directory.

Decision:

- Continue to Phase 2: Data and Baseline Smoke.
- Phase 2 should create `goal/2-baseline.md` before implementation.
- Reuse the existing venv if possible, but install the remaining Parameter Golf requirements before running `train_gpt.py`.
- Do not assume dataset download is allowed on a submit node; prepare data through a Slurm job.
