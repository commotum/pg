# Phase 2: Data and Baseline Smoke

Date drafted: 2026-06-21

## Overview

Prepare the minimal `sp1024` FineWeb cache needed by the upstream Parameter Golf baseline and run a short one-GPU training smoke using `parameter-golf/train_gpt.py`.

This phase is a signs-of-life test for the real training script, not a performance benchmark. It should prove that dependencies, data paths, tokenizer paths, CUDA execution, logging, final validation, and compressed artifact writing all work together.

## Why This Matters

The qMLP experiments are only useful if we can compare them against a working baseline under the same training harness. Phase 1 proved PyTorch can see a GPU. Phase 2 proves the actual Parameter Golf baseline can consume the cached dataset and produce the metrics we care about.

This phase moves us closer to the qMLP question by establishing the first end-to-end baseline run path.

## Assumptions and Dependencies

- Phase 0 remote repo setup is complete.
- Phase 1 PyTorch/CUDA smoke is complete.
- Remote repo path is `/nfs/hpc/share/peterj29/pg/src/pg`.
- Parameter Golf submodule path is `/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf`.
- Shared venv path is `/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621`.
- Existing venv has Python `3.11.14`, `torch 2.12.1+cu130`, `numpy 2.4.6`, and CUDA GPU execution verified.
- The remaining minimal packages needed for this phase are `huggingface-hub` for data prep and `sentencepiece` for the training script.
- `parameter-golf/.gitignore` ignores `data/datasets`, `data/tokenizers`, `data/manifest.json`, `logs/`, `*.ptz`, and model artifacts.
- The upstream baseline uses bf16 CUDA paths and flash attention, so RTX8000 is not a reliable training-smoke target. Use an A40 for the training smoke.

## Resource Requests

### Data Prep

Use CPU Slurm for data download:

```text
partition=share
time=00:30:00
nodes=1
ntasks=1
cpus-per-task=2
mem=8G
```

### Baseline Smoke

Prefer A40. The first live checks for this phase found `share/a40` to be the best fitting accessible A40 path, ahead of `ampere/a40` and `preempt/a40`:

```text
partition=share
constraint=a40
gres=gpu:1
time=00:20:00
nodes=1
ntasks=1
cpus-per-task=4
mem=32G
```

Use `ampere` if the scheduler fit is immediate or close. Use `preempt` if `ampere` is not a reasonable fit. Use `share/a40` if it is the earliest accessible A40 target and the preemption behavior is acceptable for a short smoke job.

## Implementation Steps

1. Install minimal remaining dependencies into the Phase 1 venv.

```bash
source /nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621/bin/activate
python -m pip install huggingface-hub sentencepiece tqdm
python -m pip freeze --all > /nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621.freeze.txt
```

2. Run a CPU Slurm data-prep job from the Parameter Golf submodule.

```bash
cd /nfs/hpc/share/peterj29/pg/src/pg/parameter-golf
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

This downloads the full fixed validation split plus one training shard for `sp1024`.

3. Verify the data prep job produced:

- `data/manifest.json`;
- `data/tokenizers/fineweb_1024_bpe.model`;
- at least one `data/datasets/fineweb10B_sp1024/fineweb_train_*.bin`;
- at least one `data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`.

4. Run scheduler fit checks for A40.

```bash
srun --test-only -p ampere --constraint=a40 --gres=gpu:1 \
  --time=00:20:00 --cpus-per-task=8 --mem=64G true

srun --test-only -p preempt --constraint=a40 --gres=gpu:1 \
  --time=00:20:00 --cpus-per-task=8 --mem=64G true
```

5. Submit a short baseline smoke job from a per-job working directory.

Initial smoke settings:

```text
RUN_ID=baseline_smoke_a40_job<job-id>
DATA_PATH=/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf/data/datasets/fineweb10B_sp1024
TOKENIZER_PATH=/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
VOCAB_SIZE=1024
SEED=42
ITERATIONS=2
WARMUP_STEPS=1
TRAIN_LOG_EVERY=1
VAL_LOSS_EVERY=0
TRAIN_BATCH_TOKENS=65536
VAL_BATCH_SIZE=65536
MAX_WALLCLOCK_SECONDS=0
```

`VAL_LOSS_EVERY=0` skips step-zero validation but final validation and int8+zlib roundtrip validation still run.

6. Preserve all training outputs under:

```text
/nfs/hpc/share/peterj29/pg/runs/phase2-baseline/<job-id>/
```

The job directory should include:

- `manifest.txt`;
- `gpu.txt`;
- `python.txt`;
- `train.log`;
- `logs/<RUN_ID>.txt`;
- `final_model.pt`;
- `final_model.int8.ptz`;
- `COMPLETE.txt` on success.

## Expected Artifacts

- Updated venv freeze file.
- Data prep job directory under `/nfs/hpc/share/peterj29/pg/runs/phase2-data/`.
- Baseline smoke job directory under `/nfs/hpc/share/peterj29/pg/runs/phase2-baseline/`.
- Local reusable batch scripts if created.
- This phase file updated with actual results.

## Completion Requirements

This phase is complete when:

- The minimal additional dependencies are installed in the shared venv.
- The `sp1024` tokenizer exists.
- The `sp1024` validation files exist.
- At least one `sp1024` training shard exists.
- A baseline smoke Slurm job reaches state `COMPLETED`.
- Baseline smoke exit code is `0:0`.
- Training log shows at least one completed training iteration.
- Training log includes `model_params`.
- Training log includes final `val_loss` and `val_bpb`.
- Training log includes `final_int8_zlib_roundtrip` metrics.
- `final_model.int8.ptz` exists in the preserved job directory.
- The result section below records evidence, artifacts, new facts, and the Phase 3 decision.

## Failure and Fallback Rules

- If data download fails because compute nodes cannot reach Hugging Face, rerun the data-prep command on a submit node only long enough to download the published cache, then document why the fallback was necessary.
- If full validation download is too slow for the 30-minute CPU job, preserve the partial logs and repeat with a longer CPU walltime before changing the scientific scope.
- If A40 scheduler fit is poor, run the baseline smoke on `preempt/a40` before considering H100/H200.
- If the A40 smoke OOMs, reduce `TRAIN_BATCH_TOKENS` and `VAL_BATCH_SIZE` before reducing model shape.
- If bf16 or flash attention fails on A40, capture the exact error and stop; do not hide the failure by switching to a materially different model.
- If final validation is too slow, reduce `VAL_BATCH_SIZE` only if the failure is memory-related. Do not skip final validation if the goal is to verify `val_bpb` reporting.

## Result

Status: in progress

Evidence:

- Minimal additional dependencies installed into `/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621`:
  - `huggingface-hub 1.20.1`;
  - `sentencepiece 0.2.1`;
  - `tqdm 4.68.3`.
- Data-prep Slurm job `20480484` completed:
  - partition `share`;
  - node `cn-a14`;
  - state `COMPLETED`;
  - exit code `0:0`;
  - elapsed `00:00:11`;
  - allocation `cpu=2, mem=8G, node=1`.
- Data prep ran with `HF_HOME=/nfs/hpc/share/peterj29/pg/hf-cache`.
- Data prep produced:
  - `data/manifest.json` at `1925` bytes;
  - `data/tokenizers/fineweb_1024_bpe.model` at `254483` bytes;
  - `data/tokenizers/fineweb_1024_bpe.vocab` at `9856` bytes;
  - `data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin` at `200001024` bytes;
  - `data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin` at `124044716` bytes.
- Download log warned that Hugging Face requests were unauthenticated.
- A40 scheduler checks found:
  - `ampere/a40`, 8 CPU, 64G, 20 minutes: estimated `2026-06-22T05:08:56`, with preemption;
  - `preempt/a40`, 8 CPU, 64G, 20 minutes: estimated `2026-06-22T06:09:38`;
  - `share/a40`, 4 CPU, 32G, 20 minutes: estimated `2026-06-22T00:54:18`, with preemption.
- `athena/a40` failed with `Invalid account or account/partition combination specified`.
- `all/a40` failed with `User's group not permitted to use this partition`.
- Baseline smoke job `20480529` was submitted to `share/a40` with one GPU, 4 CPUs, 32G RAM, and 20 minutes.
- Baseline smoke job `20480529` is currently pending with reason `QOSGrpCpuLimit`.
- Visible association remains `coehpc|eecs|peterj29|||normal||||`, so the job cannot be switched to another visible QOS by the agent.

Artifacts:

- Local data-prep script: `goal/2-data.sbatch`.
- Remote data-prep script: `/nfs/hpc/share/peterj29/pg/runs/phase2-data/phase2-data.sbatch`.
- Data-prep job directory: `/nfs/hpc/share/peterj29/pg/runs/phase2-data/20480484/`.
- Local baseline smoke script: `goal/2-baseline.sbatch`.
- Remote baseline smoke script: `/nfs/hpc/share/peterj29/pg/runs/phase2-baseline/phase2-baseline.sbatch`.
- Pending baseline job ID: `20480529`.

New facts:

- The published `sp1024` cache for one train shard and one validation shard is available quickly from compute nodes with `HF_HOME` on shared storage.
- The Phase 2 data step does not need to run on the submit node.
- For the current cluster state, A40 availability is gated more by account/QOS CPU limits than by script readiness.
- Using `share/a40` can still show Slurm preemption candidates; preemption is allowed when Slurm grants it, but the account QOS cap can still block job start.

Decision:

- Keep baseline smoke job `20480529` queued for now; do not submit duplicates while it is pending.
- Phase 2 remains incomplete until the baseline smoke job completes and produces `val_bpb` plus roundtrip artifact evidence.
