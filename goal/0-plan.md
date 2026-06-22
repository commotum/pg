# OSU HPC Plan for Quaternion Parameter Golf Experiments

Date drafted: 2026-06-21

## Objective

Use OSU Engineering/HPC resources to run fast, disciplined experiments on quaternion Parameter Golf ideas, with cheap/available GPUs for iteration and scarce H100/H200 resources only for confirmation.

The immediate scientific question is:

> Does replacing dense MLP projections with Hamilton-structured quaternion projections improve final Parameter Golf validation bits-per-byte after reinvesting the saved parameter budget?

The practical goal is not lower parameter count by itself. The goal is lower `val_bpb` under the challenge-style wallclock and artifact constraints.

## Operating Model

Use the OSU systems in this order:

1. Local Mac for editing, code review, planning, and git commits.
2. OSU flip/Engineering server as the gateway.
3. OSU HPC submit node for Slurm inspection, job preparation, and submission.
4. Slurm compute nodes for any training, dataset preparation, compilation, or GPU checks.

Submit nodes are control planes only. Do not run training, dataset preprocessing, GPU commands, or material tests directly on submit nodes.

Current access findings:

- Local shell alias found: `osu='ssh peterj29@access.engr.oregonstate.edu'`.
- On flip, remote alias found: `hpc='ssh peterj29@submit-a.hpc.engr.oregonstate.edu'`.
- Direct nested SSH through flip to HPC submit works.
- `codex` exists on both flip and the HPC submit node at `/nfs/stak/users/peterj29/.local/bin/codex`.
- Recommended long-session mode: SSH to the HPC submit node and start Codex from the project directory there.
- Mac-driven mode is also workable: issue explicit SSH commands through flip for inventory, `sbatch`, `squeue`, and `sacct`.

No GPU jobs should be submitted without an explicit resource request and an explicit walltime.

## Live Cluster Snapshot

Live checks were run on 2026-06-21 around 12:54 PDT through `submit-a.hpc.engr.oregonstate.edu`.

Visible Slurm association:

```text
account=eecs
default_qos=normal
```

Current user queue was empty after checks.

Relevant scheduler test-only results:

```text
gpu / rtx8000 / 1 GPU:
  starts immediately on cn-gpu5

ampere / a40 / 1 GPU:
  starts immediately on cn-r-4
  note: test-only output said it would preempt existing jobs

preempt / a40 / 1 GPU:
  estimated 2026-06-21 14:45 on cn-t-1

dgxh / h200 / 1 GPU:
  estimated 2026-06-24 00:56 on dgxh-4

dgxh / h100 / 1 GPU:
  estimated 2026-06-24 03:05 on dgxh-3
```

Interpretation:

- `gpu` with RTX8000 is the best immediate smoke-test target for your `eecs` account.
- A40 is the best development GPU class for experiments that should roughly transfer to higher-end NVIDIA hardware.
- `ampere` is stable but may preempt lower-priority work; use it when we need a real A40 result.
- `preempt` is suitable for restartable, short, cheap A40 development checks.
- H100/H200 should be reserved for final confirmation of the best candidates.
- DGX2/V100 nodes were drained/down during this snapshot and should not be part of the initial plan.

Always rerun live checks before submitting real work:

```bash
sinfo -a -o '%20P|%10a|%12l|%8D|%24F|%30G'
sinfo -a -N -o '%24N|%18P|%12T|%8c|%12m|%45G|%120f'
squeue -u "$USER"
```

For scheduler fit without submitting:

```bash
srun --test-only -p gpu --constraint=rtx8000 --gres=gpu:1 \
  --time=00:10:00 --cpus-per-task=4 --mem=16G true

srun --test-only -p ampere --constraint=a40 --gres=gpu:1 \
  --time=00:10:00 --cpus-per-task=4 --mem=16G true

srun --test-only -p preempt --constraint=a40 --gres=gpu:1 \
  --time=00:10:00 --cpus-per-task=4 --mem=16G true

srun --test-only -p dgxh --constraint=h100 --gres=gpu:1 \
  --time=00:10:00 --cpus-per-task=8 --mem=64G true
```

## Storage and Environment Plan

Use:

- Home: shell config and small source only.
- `/nfs/hpc/share/peterj29`: environments, cloned repos, datasets, run records.
- Compute-node `/scratch/$USER/$SLURM_JOB_ID`: hot training data and temporary outputs during a job.

Observed storage:

```text
$HOME: 25G filesystem, about 14G available during inspection
/nfs/hpc/share/peterj29: 1.5T filesystem, about 1.5T available during inspection
```

Do not put Python environments, FineWeb shards, checkpoints, or large logs in home.

Recommended remote layout:

```text
/nfs/hpc/share/peterj29/pg/
  src/pg/                         # git checkout
  envs/
  datasets/
  runs/
  wheels/
```

Environment approach:

1. Start with a Python venv under `hpc-share`, not home.
2. Pin exact Python, PyTorch, CUDA wheel/runtime, and dependencies in each run manifest.
3. Prefer a small PyTorch/CUDA setup for the baseline first.
4. Do not begin with the top leaderboard environment, because the strongest record stack requires FA3, PyTorch 2.9.1+cu128, CUDA 12.8, `lrzip`, and an 8xH100-oriented script.

Available modules observed:

```text
python/3.8 through python/3.13
cuda/9.2 through cuda/13.1
cudnn/8.8_cuda11
cudnn/8.9_cuda11
cudnn/8.9_cuda12
```

Default `python3` on submit was Python 3.6.8, so jobs should explicitly load a Python module or use a controlled venv.

## Agent Operating Rules

The agent may:

- inspect git state and files;
- prepare Slurm scripts;
- run live read-only inventory commands;
- submit small approved smoke tests and benchmarks;
- monitor specific job IDs;
- summarize logs and accounting;
- stage outputs into job-specific run directories.

The agent must not:

- run training or GPU diagnostics on submit nodes;
- use `sudo`;
- alter global modules or system software;
- cancel all jobs;
- delete shared data or environments without explicit approval;
- expose secrets in logs, job names, `#SBATCH` lines, command arguments, or full environment dumps;
- request more than 1 GPU, more than 16 CPUs, more than 128G RAM, or more than 8 hours without explicit human approval.

For normal development, keep jobs at:

```text
nodes: 1
gpus: 1
cpus-per-task: 4-8
memory: 16G-64G
walltime: 10-30 minutes for smoke/bench
```

Use `sbatch`, not long attached `srun`, for unattended runs.

## Why Not Start With the Best Leaderboard Stack

The best record in the local submodule, `2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611`, is not a good first A40 target.

It assumes:

- 8x H100 SXM;
- 600-second wallclock with about 4,900 training steps;
- FlashAttention 3;
- PyTorch 2.9.1+cu128;
- CUDA 12.8;
- `lrzip` system binary;
- large custom `train_gpt.py` with many stacked architectural ideas;
- SP8192 CaseOps tokenizer and sidecar data;
- phased TTT evaluation.

That stack is good for final comparison, but it is too complex for first validation of quaternion MLP layers. First prove the layer idea on the simpler baseline.

## Scientific Strategy

Quaternion MLP saves parameters in the Transformer blocks. For the default simple baseline:

```text
NUM_LAYERS=9
MODEL_DIM=512
MLP_MULT=2
hidden=1024
```

Each MLP block has:

```text
fc:   512 * 1024 = 524,288
proj: 1024 * 512 = 524,288
total per block = 1,048,576
```

A quaternion MLP uses one quarter as many learned matrix parameters:

```text
quaternion MLP per block = 262,144
saved per block = 786,432
saved across 9 blocks = 7,077,888
```

This is a large parameter saving, but Parameter Golf is about `val_bpb`, not parameter count. Therefore the right use of qMLP is to buy budget for changes that reduce BPB:

1. larger vocabulary/tokenizer;
2. modest width increase;
3. modest depth increase;
4. stronger record-stack integration only after qMLP proves useful.

Current hypothesis:

> qMLP plus larger vocab/tokenizer is more promising than qMLP plus maximum width/depth, because better tokenization attacks BPB more directly while qMLP offsets the embedding/head parameter cost.

## Implementation Plan

### Phase 0: Remote Project Setup

Goal: create a clean remote working copy and run inventory.

Actions:

1. On HPC submit node, create project root:

```bash
mkdir -p /nfs/hpc/share/$USER/pg/src
cd /nfs/hpc/share/$USER/pg/src
```

2. Clone or update this repo:

```bash
git clone git@github.com:commotum/pg.git pg
cd pg
git checkout mac
git submodule update --init --recursive
```

3. Run inventory from the guide:

```bash
cd /nfs/hpc/share/$USER/pg/src/pg/osu-hpc-agent-guide
bash bin/osu-hpc-inventory.sh
```

4. Preserve inventory under the run tree or commit a summary, not the whole raw inventory unless it is intentionally tracked.

Exit criteria:

- repo exists on `hpc-share`;
- submodules are checked out;
- live inventory is captured;
- no compute work has run on submit.

### Phase 1: Environment Smoke Test

Goal: verify Python, CUDA, PyTorch, and GPU visibility inside Slurm.

Preferred first target:

```text
partition=gpu
constraint=rtx8000
gres=gpu:1
time=00:10:00
cpus=4
mem=16G
```

Why:

- schedulable immediately in live test;
- available to `eecs`;
- lower-stakes than A40/H100/H200;
- good enough to verify environment and job mechanics.

Job should record:

- hostname;
- `SLURM_*` variables;
- `CUDA_VISIBLE_DEVICES`;
- `nvidia-smi -L`;
- `nvidia-smi`;
- module list;
- Python version;
- PyTorch version;
- `torch.cuda.is_available()`;
- GPU name from PyTorch.

Exit criteria:

- Slurm state `COMPLETED`;
- exit code `0:0`;
- PyTorch sees exactly the allocated GPU;
- manifest and logs are preserved under `/nfs/hpc/share/$USER/pg/runs/...`.

### Phase 2: Data and Baseline Smoke

Goal: get the simplest Parameter Golf baseline running on one GPU with a tiny workload.

Use the upstream `parameter-golf/train_gpt.py`, not a record script.

Start with small data prep if required:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

If dataset download is large or slow, run it as a CPU Slurm job or a short GPU-independent job. Do not run it directly on submit.

Initial smoke settings:

```text
RUN_ID=baseline_smoke
ITERATIONS=20
MAX_WALLCLOCK_SECONDS=0
VAL_LOSS_EVERY=0
TRAIN_BATCH_TOKENS=65536
VAL_BATCH_SIZE=65536
```

Exit criteria:

- training script starts;
- one or more iterations complete;
- logs contain ms/step;
- no GPU OOM;
- no missing data/module/dependency issues.

### Phase 3: A40 Baseline Benchmark

Goal: establish the local A40 speed and BPB baseline.

Target:

```text
partition=ampere or preempt
constraint=a40
gres=gpu:1
time=00:15:00
cpus=8
mem=64G
```

Use `ampere` when we need a stable non-preempt comparison and the queue fit is acceptable. Use `preempt` for cheaper iteration, but make runs short and restartable.

Run settings:

```text
RUN_ID=baseline_a40_10m_seed42
SEED=42
MAX_WALLCLOCK_SECONDS=600
ITERATIONS=20000
VAL_LOSS_EVERY=0
TRAIN_BATCH_TOKENS=<calibrate for A40>
VAL_BATCH_SIZE=<small enough to fit>
```

Capture:

- steps completed in 600s;
- ms/step median and mean;
- final `val_loss`;
- final `val_bpb`;
- max GPU memory;
- GPU utilization snapshot if available;
- artifact size if the script emits one;
- `sacct` accounting.

Exit criteria:

- completed 10-minute run;
- known A40 steps-per-10-min baseline;
- known A40 `val_bpb` baseline;
- run manifest preserved.

### Phase 4: Quaternion MLP Correctness

Goal: implement and test `QuaternionLinear` for MLP replacement without changing attention or tokenizer.

First implementation should:

- store component weights as separate 2D parameters, not one 3D tensor;
- preserve existing Muon optimizer grouping where possible;
- require input and output dimensions divisible by 4;
- use a feature flag, e.g. `QUAT_MLP=1`;
- leave Q/K/V and fused attention unchanged.

Reason for separate 2D parameters:

`train_gpt.py` currently sends only `p.ndim == 2` block params to Muon. A single `[4, out_q, in_q]` parameter would be missed unless optimizer grouping is changed intentionally.

Correctness tests:

- shape preservation for `512 -> 1024` and `1024 -> 512`;
- gradients exist for all four component matrices;
- small numerical check against explicit Hamilton block matrix;
- model forward on a toy batch;
- one short training smoke run.

Exit criteria:

- qMLP smoke run completes;
- no optimizer parameter is accidentally excluded;
- parameter count reduction matches expectation;
- compile path does not fail under `torch.compile`.

### Phase 5: qMLP A40 Benchmark

Goal: compare qMLP with the baseline under the same A40 wallclock.

Run:

```text
RUN_ID=qmlp_a40_10m_seed42
SEED=42
QUAT_MLP=1
MAX_WALLCLOCK_SECONDS=600
ITERATIONS=20000
VAL_LOSS_EVERY=0
same train/eval batch settings as baseline
```

Compare against Phase 3:

- steps completed;
- ms/step;
- `val_bpb`;
- GPU memory;
- artifact size/parameter count;
- training stability.

Decision gate:

- If qMLP is much slower and worse BPB at same config, stop and optimize implementation before reinvesting saved params.
- If qMLP is close in BPB and not dramatically slower, continue to vocabulary/width/depth reinvestment.
- If qMLP improves BPB directly, prioritize vocab and seed replication.

### Phase 6: Reinvestment Grid

Goal: spend qMLP's saved parameter budget where it most improves BPB.

Initial grid:

```text
baseline:
  512d / 9L / vocab1024

qmlp:
  512d / 9L / vocab1024
  512d / 9L / vocab4096
  512d / 9L / vocab8192
  576d / 9L / vocab4096
  640d / 9L / vocab4096
  512d / 12L / vocab4096
```

Do not jump immediately to same-parameter maximum width/depth. Width and depth increase FLOPs and may reduce steps in 600s. Vocabulary may improve BPB more directly.

For each candidate:

- run a short smoke first;
- run one 10-minute A40 benchmark;
- log seed, shape, vocab, tokenizer path, steps, ms/step, `val_bpb`, and memory.

Decision gate:

- Promote candidates that improve `val_bpb` at 600s, not candidates that merely reduce parameters.
- Penalize candidates that improve fixed-step loss but lose too many wallclock steps.
- Prefer candidates that remain simple enough to port into record stacks.

### Phase 7: Seed Replication

Goal: avoid chasing noise.

For the best 2-3 A40 candidates, run:

```text
SEED=42
SEED=0
SEED=1234
```

Use the same GPU class where possible. If scheduling forces mixed A40 nodes, record hostnames and GPU names.

Promote only if:

- mean BPB improves over baseline;
- no seed catastrophically regresses;
- speed is acceptable;
- artifact size remains within the challenge budget path.

### Phase 8: H100/H200 Confirmation

Goal: test the best candidate on scarce premium hardware.

Target:

```text
partition=dgxh
constraint=h100 or h200
gres=gpu:1 initially
time=00:15:00 for single-GPU scaling check
```

Do not request 8 GPUs until a 1-GPU H-class run confirms:

- environment compatibility;
- no architecture-specific crash;
- expected speedup over A40;
- memory fit.

Then, if justified and approved:

```text
partition=dgxh
constraint=h100 or h200
gres=gpu:8
time=00:15:00 to 00:30:00 for pilot
```

Final record-style run should use the actual competition-like resource shape only after:

- code path is stable;
- dataset is staged;
- logs are clean;
- exact command is reviewed;
- expected cost/fairness impact is acceptable.

### Phase 9: Integration Into Strong Record Stack

Goal: determine whether qMLP still helps when stacked with stronger known tricks.

Only after the simple baseline proves qMLP is useful:

1. choose a strong but manageable record script, not necessarily the absolute top one;
2. port qMLP into its MLP layer;
3. run a fixed-step sanity check;
4. run A40 short wallclock comparison;
5. run H-class confirmation if promising.

Avoid porting into the full 1.061 stack first. Too many confounders:

- tokenizer changes;
- CaseOps;
- sparse attention gate;
- SmearGate;
- LQER;
- GPTQ;
- compression;
- phased TTT;
- FA3.

## Benchmark Report Template

Every run should produce a small summary:

```text
run_id:
git_commit:
host:
partition:
gpu_name:
seed:
model_dim:
layers:
vocab_size:
quat_mlp:
quat_attn:
train_batch_tokens:
max_wallclock_seconds:
steps_completed:
ms_per_step_mean:
ms_per_step_median:
val_loss:
val_bpb:
artifact_size:
max_gpu_mem:
slurm_job_id:
slurm_state:
exit_code:
notes:
```

Store summaries under:

```text
/nfs/hpc/share/$USER/pg/runs/<run_id>/summary.txt
```

Optionally copy curated summaries back into this repo under:

```text
knowledge/runs/
```

Do not commit giant logs or datasets.

## Initial Job Resource Recommendations

RTX8000 smoke:

```text
partition=gpu
constraint=rtx8000
gres=gpu:1
time=00:10:00
cpus-per-task=4
mem=16G
```

A40 short smoke:

```text
partition=preempt
constraint=a40
gres=gpu:1
time=00:10:00
cpus-per-task=4
mem=16G
```

A40 10-minute benchmark:

```text
partition=ampere
constraint=a40
gres=gpu:1
time=00:15:00
cpus-per-task=8
mem=64G
```

H100/H200 confirmation:

```text
partition=dgxh
constraint=h100 or h200
gres=gpu:1
time=00:15:00
cpus-per-task=8
mem=64G
```

Only request 8x H100/H200 after explicit approval.

## Expected Bottlenecks

Scheduler:

- A40 nodes are often occupied by CPU-heavy jobs that may not show GPU use in `AllocTRES`, but CPU exhaustion can still block placement.
- `ampere` can start immediately but may preempt lower-priority jobs.
- `preempt` is lower priority and may wait longer or be interrupted.
- H100/H200 are scarce and currently estimate days out.

Software:

- default submit-node Python is too old;
- PyTorch/CUDA must be pinned;
- top record scripts may require FA3 wheels not available by default;
- custom kernels may behave differently on A40 versus H100.

Modeling:

- qMLP saves parameters but may reduce expressiveness;
- naive qMLP implementation may be slower despite fewer weights;
- larger vocab improves BPB only if tokenizer/data prep is correct;
- width/depth may improve fixed-step quality but lose wallclock steps.

Filesystem:

- FineWeb shards and environments should live on `hpc-share`;
- hot training data should be staged to local `/scratch`;
- avoid many tiny files and repeated environment activation.

## Near-Term To-Do List

1. Start Codex on HPC submit node from `/nfs/hpc/share/$USER/pg/src/pg`.
2. Run `osu-hpc-agent-guide/bin/osu-hpc-inventory.sh`.
3. Create an HPC-specific `scripts/` directory for Parameter Golf Slurm jobs.
4. Create a GPU smoke sbatch derived from `osu-hpc-agent-guide/templates/gpu.sbatch`.
5. Create a baseline Parameter Golf sbatch for one-GPU smoke.
6. Build a minimal Python/PyTorch environment under `/nfs/hpc/share/$USER/pg/envs`.
7. Download or stage the smallest needed FineWeb/tokenizer data.
8. Run RTX8000 environment smoke.
9. Run A40 baseline 10-minute benchmark.
10. Implement qMLP locally and push branch.
11. Pull qMLP branch on HPC.
12. Run qMLP smoke and A40 10-minute benchmark.
13. Decide whether to reinvest saved params into vocab, width, or depth based on measured BPB and speed.

## First Success Definition

The first milestone is complete when we have:

- a reproducible A40 baseline 10-minute run;
- a reproducible qMLP A40 10-minute run;
- both run summaries preserved;
- a clear comparison of `val_bpb`, steps, ms/step, memory, and parameter/artifact size;
- a decision about whether qMLP is worth reinvesting into larger vocab/width/depth.
