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

Phase 0 inventory was later captured on 2026-06-21 at 22:26 PDT through `submit-a.ib.coehpc` and preserved under:

```text
/nfs/hpc/share/peterj29/pg/runs/inventory/20260621-222534/
```

That later inventory confirmed the visible Slurm association was still `coehpc|eecs|peterj29|||normal||||`, the user queue was empty, `/nfs/hpc/share/peterj29` had about 1.5T available, and submit-node GPU runtime inspection was intentionally skipped because no Slurm GPU allocation was requested.

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

Phase 0 result as of 2026-06-21 22:26 PDT:

- remote repo exists at `/nfs/hpc/share/peterj29/pg/src/pg`;
- remote branch is `mac` at `e45bef8afd9c4129850aae14f4d0c1fd8543fbad`;
- remote source tree is clean after moving generated inventory output into the run tree;
- `parameter-golf` is checked out at `f5c079314c4877fbb0af378c0abade5a8ca33d3a`;
- `qham` is checked out at `fb7b546294aecdabace2f5fab0527001df320b77`;
- full inventory artifact is under `/nfs/hpc/share/peterj29/pg/runs/inventory/20260621-222534/`;
- no compute work was run on the submit node.

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

Phase 1 result as of 2026-06-21 22:40 PDT:

- shared venv exists at `/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621`;
- `pip install torch numpy` resolved to `torch 2.12.1+cu130`, `torch.version.cuda == 13.0`, and `numpy 2.4.6`;
- smoke job `20480372` ran on `cn-gpu7` in partition `gpu` with `constraint=rtx8000`, `gres/gpu:1`, 4 CPUs, 16G RAM, and 10 minute walltime;
- Slurm state was `COMPLETED`, exit code `0:0`, elapsed `00:00:20`;
- `nvidia-smi -L` saw exactly one `Quadro RTX 8000` with 46080 MiB;
- job loaded `slurm/current` and `cuda/13.0`;
- PyTorch saw exactly one CUDA device and completed a small CUDA forward/backward pass;
- artifacts are under `/nfs/hpc/share/peterj29/pg/runs/phase1-smoke/20480372/`;
- local reusable batch script is `goal/1-smoke.sbatch`.

Phase 2 should reuse this venv if practical, but it still needs the remaining Parameter Golf dependencies and data preparation through Slurm.

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

Phase 2 result as of 2026-06-22 01:07 PDT:

- venv `/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621` was extended with `huggingface-hub`, `sentencepiece`, and `tqdm`;
- CPU data-prep job `20480484` ran on `share`/`cn-a14`, completed with exit code `0:0`, and used `HF_HOME=/nfs/hpc/share/peterj29/pg/hf-cache`;
- data prep produced one `sp1024` train shard, one validation shard, manifest, and tokenizer files under the Parameter Golf submodule;
- live A40 checks found `share/a40` was the earliest accessible A40 path, while `athena/a40` and `all/a40` were not permitted for this account;
- baseline smoke job `20480529` ran on `share/a40`/`cn-r-5` with one A40, 4 CPUs, 32G RAM, and 20 minute walltime;
- job `20480529` initially pending on `QOSGrpCpuLimit`, then completed with state `COMPLETED`, exit code `0:0`, elapsed `00:03:43`;
- training settings were `ITERATIONS=2`, `WARMUP_STEPS=1`, `TRAIN_BATCH_TOKENS=65536`, `VAL_BATCH_SIZE=65536`, and `MAX_WALLCLOCK_SECONDS=0`;
- baseline smoke reported `model_params:17059912`, step average about `262ms` for the two measured training steps, final `val_bpb:4.1008`, and int8+zlib roundtrip `val_bpb:4.10409907`;
- final int8+zlib artifact size was `4963374` bytes, with total int8+zlib submission size `5011060` bytes;
- artifacts are under `/nfs/hpc/share/peterj29/pg/runs/phase2-baseline/20480529/`;
- this was a correctness smoke, not a meaningful model-quality benchmark.

### Phase 3: A40 Baseline Benchmark

Goal: establish the local A40 speed and BPB baseline.

Target:

```text
partition=share
constraint=a40
gres=gpu:1
time=00:25:00
cpus=2
mem=24G
```

Use `share/a40` when it is the earliest accessible A40 path. Use `ampere` when we need a stable non-preempt comparison and the queue fit is acceptable. Use `preempt` for cheaper iteration, but make runs short and restartable.

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

Phase 3 result as of 2026-06-22 01:40 PDT:

- completed default `sp1024` data cache with 80 train shards and one validation shard via CPU Slurm job `20480545`;
- best current A40 path was `share/a40`; `ampere/a40` and `preempt/a40` were later fits;
- account/QOS pressure made lower CPU requests important: 64G and 32G benchmark attempts were canceled while pending on `QOSGrpCpuLimit`, then the successful benchmark used 2 CPUs and 24G RAM;
- A40 benchmark job `20480569` ran on `cn-r-3`, partition `share`, one `NVIDIA A40`, 2 CPUs, 24G RAM, and 25 minute Slurm walltime;
- job `20480569` completed with state `COMPLETED`, exit code `0:0`, elapsed `00:14:26`;
- run config used the upstream baseline with `TRAIN_BATCH_TOKENS=524288`, `VAL_BATCH_SIZE=524288`, `WARMUP_STEPS=20`, `MAX_WALLCLOCK_SECONDS=600`, `SEED=42`;
- baseline reached `step:379/20000` with measured train time `601503ms` and step average `1587.08ms`;
- final validation before quantized roundtrip was `val_loss:2.6312`, `val_bpb:1.5584`;
- peak memory allocated was `13129 MiB`, reserved `13194 MiB`;
- fp32 model size was `67224983` bytes;
- int8+zlib artifact was `9265169` bytes, total int8+zlib submission size `9312855` bytes;
- roundtrip metrics were `val_loss:2.66913307`, `val_bpb:1.58081095`, eval time `55294ms`;
- artifacts are under `/nfs/hpc/share/peterj29/pg/runs/phase3-a40/20480569/`.

Phase 4 should use this as the first qMLP comparison reference, especially `model_params:17059912`, `1.587s/step`, `379` steps in 600 seconds, and roundtrip `val_bpb:1.58081095`.

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

Phase 4 result as of 2026-06-22 02:02 PDT:

- implemented `QUAT_MLP=1` in `parameter-golf/train_gpt.py` as a feature-flagged MLP-only replacement;
- `QUAT_MLP=0` preserves the dense `CastedLinear` path;
- qMLP uses separate 2D `wr`, `wi`, `wj`, and `wk` parameters per quaternion projection, keeping the existing Muon matrix-parameter grouping valid;
- local syntax check passed with `python3 -m py_compile parameter-golf/train_gpt.py`;
- remote correctness script passed shape checks, Hamilton equivalence with max error `1.19e-07`, gradient/forward-backward checks, and parameter counting;
- qMLP model parameters are `9982024` versus dense baseline `17059912`, saving `7077888` parameters;
- qMLP component matrices counted for Muon grouping: `72`;
- the initial 2-CPU smoke job `20480593` was canceled while pending on `QOSGrpCpuLimit`; a 1-CPU / 16G A40 request was schedulable immediately and sufficient for the smoke;
- qMLP smoke job `20480598` ran on `cn-r-4`, partition `share`, one `NVIDIA A40`, 1 CPU, 16G RAM, and 20 minute Slurm walltime;
- job `20480598` completed with state `COMPLETED`, exit code `0:0`, elapsed `00:07:18`;
- qMLP smoke ran `ITERATIONS=2`, `WARMUP_STEPS=1`, `TRAIN_BATCH_TOKENS=65536`, `VAL_BATCH_SIZE=65536`, `SEED=42`;
- qMLP smoke reached `step:2/2`, final `val_loss:6.9227`, `val_bpb:4.1000`, and `step_avg:401.60ms`;
- qMLP smoke roundtrip metrics were `val_loss:6.92820268`, `val_bpb:4.10327187`, eval time `114807ms`;
- qMLP smoke peak memory was `1796 MiB` allocated and `1834 MiB` reserved;
- qMLP fp32 model size was `38930363` bytes;
- qMLP int8+zlib artifact was `6050417` bytes, total int8+zlib submission size `6100622` bytes;
- artifacts are under `/nfs/hpc/share/peterj29/pg/runs/phase4-qmlp/20480598/`;
- the qMLP implementation is correct enough for a same-wallclock Phase 5 benchmark, but its tiny-smoke speed is slower than dense and must be measured under the 600-second benchmark before reinvesting saved parameters.

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

Phase 5 result as of 2026-06-22 02:26 PDT:

- exact Phase 3 resource shape was used: `share/a40`, one `NVIDIA A40`, 2 CPUs, 24G RAM, and 25 minute Slurm walltime;
- qMLP benchmark job `20480606` ran on `cn-r-4` and completed with state `COMPLETED`, exit code `0:0`, elapsed `00:17:51`;
- qMLP benchmark used `TRAIN_BATCH_TOKENS=524288`, `VAL_BATCH_SIZE=524288`, `WARMUP_STEPS=20`, `MAX_WALLCLOCK_SECONDS=600`, `SEED=42`, and `QUAT_MLP=1`;
- qMLP model parameters were `9982024`, saving `7077888` parameters versus dense baseline `17059912`;
- qMLP reached `263` steps under the 600-second training cap versus dense baseline `379` steps;
- qMLP step average was `2283.03ms` versus dense baseline `1587.08ms`, about `1.44x` slower;
- qMLP final validation before quantized roundtrip was `val_loss:3.1467`, `val_bpb:1.8637`;
- qMLP roundtrip metrics were `val_loss:3.16134623`, `val_bpb:1.87232731`, eval time `103987ms`;
- dense baseline roundtrip was `val_bpb:1.58081095`, so qMLP was worse by about `0.2915` BPB;
- qMLP peak memory was `13449 MiB` allocated and `13566 MiB` reserved, slightly higher than dense baseline;
- qMLP int8+zlib artifact was `8332875` bytes, about `0.93 MB` smaller than dense baseline but not enough to offset the BPB loss;
- artifacts are under `/nfs/hpc/share/peterj29/pg/runs/phase5-qmlp/20480606/`;
- decision: do not proceed directly to vocabulary/width/depth reinvestment. Insert an optimization phase first.

### Phase 6: qMLP Implementation Optimization

Goal: determine whether the qMLP failure is mainly implementation inefficiency or an inherent architecture/optimization problem.

The naive qMLP implementation expands each quaternion projection into many `F.linear` calls. This preserves parameter count but likely hurts throughput and eval time.

Next implementation should test one or more faster equivalent forms without changing the learned degrees of freedom:

1. materialize the equivalent Hamilton block matrix in a layout that lowers through fewer matmuls;
2. try a packed component layout that uses grouped/batched matmul instead of repeated small `F.linear` calls;
3. preserve separate learned `wr`, `wi`, `wj`, and `wk` parameter tensors if that remains the simplest way to keep optimizer grouping correct;
4. keep tokenizer, attention, depth, width, batch size, and quantization unchanged.

Exit criteria:

- optimized qMLP correctness checks pass against the same Hamilton-equivalence test;
- optimized qMLP smoke completes;
- optimized qMLP 10-minute A40 benchmark is rerun against Phase 3;
- decision says whether qMLP is fast enough to re-open reinvestment.

Decision gate:

- If optimized qMLP remains much slower and worse BPB, stop the simple qMLP path.
- If optimized qMLP becomes close enough in speed/BPB, continue to the reinvestment grid.
- If optimized qMLP improves BPB directly, prioritize vocabulary and seed replication.

Phase 6 partial result as of 2026-06-22 03:19 PDT:

- added `QUAT_MLP_IMPL` with `split` and `matrix` qMLP implementations;
- `split` preserves the Phase 5 repeated-component-matmul path;
- `matrix` constructs the equivalent Hamilton block matrix and runs one `F.linear` per quaternion projection;
- both implementations preserve the same learned `wr`, `wi`, `wj`, and `wk` tensors;
- local syntax checks passed for `parameter-golf/train_gpt.py`, `goal/4-qmlp-check.py`, `goal/6-smoke.sbatch`, and `goal/6-benchmark.sbatch`;
- remote CPU correctness check passed with `matrix_split_equivalence_max_error=1.19e-07`, `qmlp_params=9982024`, `saved_params=7077888`, and `qmlp_muon_matrix_params=72`;
- remote training and correctness files were staged under `/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf/`;
- Phase 6 smoke and benchmark scripts were staged under `/nfs/hpc/share/peterj29/pg/runs/phase6-matrix-smoke/` and `/nfs/hpc/share/peterj29/pg/runs/phase6-matrix-benchmark/`;
- A40 smoke job `20480617` was submitted but stayed pending on `QOSGrpCpuLimit` with no start estimate;
- `preempt/a40` and `ampere/a40` were later than `share/a40`, so switching partitions was not useful;
- job `20480617` was canceled while pending;
- matrix smoke job `20480622` later ran on `cn-r-4`, partition `share`, one `NVIDIA A40`, 1 CPU, 16G RAM, and completed with state `COMPLETED`, exit code `0:0`, elapsed `00:04:18`;
- matrix smoke reached `step:2/2`, `step_avg:307.63ms`, final `val_bpb:4.1000`, roundtrip `val_bpb:4.10327189`, peak memory `1748 MiB`, and int8+zlib artifact `6050224` bytes;
- exact 2-CPU/24G benchmark jobs `20480631` and `20480651` were each submitted after favorable test-only checks, but both stayed pending on `QOSGrpCpuLimit` with no start estimate and were canceled;
- provisional matrix benchmark job `20480636` ran on `cn-r-3`, partition `share`, one `NVIDIA A40`, 1 CPU, 16G RAM, and completed with state `COMPLETED`, exit code `0:0`, elapsed `00:15:07`;
- provisional matrix benchmark reached `368` steps in `600140ms`, `step_avg:1630.81ms`, final `val_bpb:1.6420`, roundtrip `val_bpb:1.64863035`, peak memory `13047 MiB`, and int8+zlib artifact `8733639` bytes;
- comparison: dense Phase 3 used 2 CPUs/24G and reached `379` steps, `step_avg:1587.08ms`, roundtrip `val_bpb:1.58081095`; split qMLP Phase 5 used 2 CPUs/24G and reached `263` steps, `step_avg:2283.03ms`, roundtrip `val_bpb:1.87232731`;
- interpretation: matrix qMLP fixes most of the split implementation speed problem, but the provisional result still trails dense by about `0.0678` roundtrip BPB;
- decision: Phase 6 is complete for decision-making. The exact 2-CPU/24G matrix rerun remains useful bookkeeping, but the provisional run is enough to stop looping on `QOSGrpCpuLimit` and move to the original saved-budget question. Proceed to a narrow vocabulary reinvestment phase before any broad width/depth grid.

### Phase 7: Reinvestment Grid

Goal: spend qMLP's saved parameter budget where it most improves BPB.

Revised after Phase 6:

- Start with vocabulary reinvestment, not a broad width/depth grid.
- The published cached manifest currently exposes only `sp1024`, so larger vocabulary experiments require rebuilding tokenizer/data from the published docs cache.
- Treat `sp4096` as the first candidate because it uses about `1.57M` extra tied embedding parameters relative to `sp1024`, well inside qMLP's `7.08M` saved parameters.
- Keep `sp8192` as a second candidate only if `sp4096` improves BPB enough to justify the export cost.

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

Phase 7 progress as of 2026-06-22 03:52 PDT:

- added `goal/7-vocab.md` as the narrow `sp4096` reinvestment phase;
- patched `data/download_hf_docs_and_tokenize.py` with `--max-train-shards` so retokenized exports can stop after full validation plus 80 train shards;
- added `goal/7-sp4096-tokenizer-config.json`, `goal/7-docs.sbatch`, and `goal/7-data.sbatch`;
- remote cached manifest currently exposes only `sp1024`, so `sp4096` requires retokenizing from selected docs;
- docs materialization job `20480690` completed on `cn-a26`, partition `share`, 1 CPU, 16G RAM, elapsed `00:08:55`;
- materialized `/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf/data/docs_selected.jsonl` at `45G` and sidecar at `481` bytes;
- docs SHA256 from the sidecar and file check is `84386dfa7b339a5d4831d5273c4a2028b78b60670d3a235633a8520545d19bc7`;
- sidecar reports `num_docs=15368808`, `docs_val=50000`, `docs_train=15318808`, and `selection_seed=1337`;
- first full `sp4096` export attempts, jobs `20480667` and `20480671`, were canceled while pending on `QOSGrpCpuLimit`;
- bounded `sp4096` export job `20480717` completed on `cn-a14`, partition `share`, 2 CPUs, 24G RAM, elapsed `02:10:25`;
- export wrote `/nfs/hpc/share/peterj29/pg/data-exports/sp4096-80/` with tokenizer `sp_bpe_4096`, vocab size `4096`, 80 train shards, one validation shard, `8000000433` train tokens, and `45517764` validation tokens;
- matrix qMLP `sp4096` smoke job `20480883` completed on `cn-r-5`, partition `share`, one A40, 1 CPU, 16G RAM, elapsed `00:04:13`;
- `sp4096` smoke reported `model_params:11554888`, `step_avg:319.65ms`, roundtrip `val_bpb:3.61143104`, and int8+zlib artifact `6403349` bytes;
- matrix qMLP `sp4096` benchmark job `20480898` completed on `cn-r-3`, partition `share`, one A40, 2 CPUs, 24G RAM, elapsed `00:14:25`;
- `sp4096` benchmark reached `352` steps in `601019ms`, `step_avg:1707.44ms`, final `val_bpb:1.5470`, roundtrip `val_bpb:1.55222627`, peak memory `13443 MiB`, and int8+zlib artifact `9931222` bytes;
- comparison: dense `sp1024` Phase 3 roundtrip was `1.58081095`, so qMLP matrix `sp4096` improved roundtrip BPB by about `0.0286` despite completing `27` fewer steps;
- comparison: qMLP matrix `sp1024` Phase 6 provisional roundtrip was `1.64863035`, so vocabulary reinvestment improved qMLP by about `0.0964` BPB;
- decision: this is a positive answer for the core qMLP reinvestment hypothesis. Proceed to Phase 8 seed replication of the `sp4096` candidate before trying `sp8192` or width/depth.

### Phase 8: Seed Replication

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

Phase 8 result as of 2026-06-22 07:28 PDT:

- added `goal/8-seed.md` and `goal/8-seed.sbatch` for one-seed-at-a-time replication of the Phase 7 `sp4096` matrix qMLP candidate;
- seed `0` benchmark job `20480958` ran on `cn-r-3`, partition `share`, one A40, 2 CPUs, 24G RAM, and completed with state `COMPLETED`, exit code `0:0`, elapsed `00:12:44`;
- seed `1` benchmark job `20480988` ran on `cn-r-6`, partition `share`, one A40, 2 CPUs, 24G RAM, and completed with state `COMPLETED`, exit code `0:0`, elapsed `00:14:25`;
- seed `0` used `QUAT_MLP=1`, `QUAT_MLP_IMPL=matrix`, `VOCAB_SIZE=4096`, `TRAIN_BATCH_TOKENS=524288`, `VAL_BATCH_SIZE=524288`, `MAX_WALLCLOCK_SECONDS=600`, and `SEED=0`;
- seed `1` used the same shape with `SEED=1`;
- seed `0` reached `353` steps in `600379ms`, `step_avg:1700.79ms`, final `val_bpb:1.5419`, roundtrip `val_bpb:1.54759284`, peak memory `13443 MiB`, and int8+zlib artifact `9948374` bytes;
- seed `1` reached `352` steps in `601374ms`, `step_avg:1708.45ms`, final `val_bpb:1.5534`, roundtrip `val_bpb:1.55872027`, peak memory `13443 MiB`, and int8+zlib artifact `9942683` bytes;
- comparison: dense `sp1024` Phase 3 roundtrip was `1.58081095`, so seed `0` improved roundtrip BPB by about `0.0332`;
- comparison: dense `sp1024` Phase 3 roundtrip was `1.58081095`, so seed `1` improved roundtrip BPB by about `0.0221`;
- comparison: across seed `42`, seed `0`, and seed `1`, qMLP `sp4096` mean roundtrip BPB is about `1.55284646`, an average improvement of about `0.0280` BPB over dense `sp1024`;
- decision: Phase 8 is complete. Promote `sp4096` matrix qMLP as the current best simple candidate and test `sp8192` next before width/depth or H100/H200 confirmation.

### Phase 9: sp8192 Vocabulary Probe

Goal: test whether the validated qMLP saved-parameter budget can support a larger `sp8192` tokenizer and improve BPB beyond the replicated `sp4096` result.

Revised after Phase 8:

- The relevant local record track is `track_10min_16mb`; `sp4096` being near 10 MB is acceptable but still needs artifact tracking.
- `sp8192` adds `2,097,152` tied embedding parameters relative to `sp4096`.
- Predicted qMLP `sp8192` model size is `13,652,040` parameters, still below dense `sp1024` by `3,407,872` parameters.
- Artifact size is plausible under 16 MB, but must be measured before a full benchmark.

Run sequence:

1. Export bounded `sp8192` tokenizer/data with 80 train shards and one validation shard.
2. Run a two-step A40 smoke with `VOCAB_SIZE=8192`.
3. Stop if smoke total int8+zlib submission size is at or above 16 MB.
4. If the artifact-size gate passes, run one seed `42` 10-minute A40 benchmark.
5. Compare against `sp4096` seed `42` and the Phase 8 `sp4096` three-run mean.

Decision gate:

- If `sp8192` improves BPB and stays under 16 MB, seed-replicate it before H100/H200.
- If `sp8192` loses to `sp4096`, keep `sp4096` as the simple candidate and consider width/depth or stronger-stack integration.
- If `sp8192` exceeds 16 MB, stop the larger-vocab path unless compression changes become the explicit next phase.

Phase 9 files:

- `goal/9-sp8192.md`
- `goal/9-sp8192-tokenizer-config.json`
- `goal/9-data.sbatch`
- `goal/9-smoke.sbatch`
- `goal/9-benchmark.sbatch`

Phase 9 result as of 2026-06-22 10:31 PDT:

- added `goal/9-sp8192.md`, `goal/9-sp8192-tokenizer-config.json`, `goal/9-data.sbatch`, `goal/9-smoke.sbatch`, and `goal/9-benchmark.sbatch`;
- CPU export job `20481014` completed on `cn-a26`, partition `share`, 2 CPUs, 24G RAM, elapsed `02:27:04`;
- export wrote `/nfs/hpc/share/peterj29/pg/data-exports/sp8192-80/` with tokenizer `sp_bpe_8192`, vocab size `8192`, 80 train shards, one validation shard, `8000009615` train tokens, and `40542913` validation tokens;
- A40 smoke job `20482322` completed on `cn-r-2`, partition `share`, one A40, 1 CPU, 16G RAM, elapsed `00:03:57`;
- `sp8192` smoke reported `model_params:13652040`, `step_avg:327.54ms`, roundtrip `val_bpb:3.48484683`, and total int8+zlib submission size `6942514` bytes;
- smoke passed the 16 MB artifact-size gate;
- A40 benchmark job `20482977` completed on `cn-r-1`, partition `share`, one A40, 2 CPUs, 24G RAM, elapsed `00:14:29`;
- `sp8192` benchmark reached `336` steps in `601192ms`, `step_avg:1789.26ms`, final `val_bpb:1.5204`, roundtrip `val_bpb:1.52530269`, peak memory `13971 MiB`, and total int8+zlib submission size `11602814` bytes;
- comparison: dense `sp1024` Phase 3 roundtrip was `1.58081095`, so `sp8192` improved roundtrip BPB by about `0.0555`;
- comparison: Phase 8 `sp4096` three-run mean was `1.55284646`, so `sp8192` improved roundtrip BPB by about `0.0275`;
- decision: promote `sp8192` matrix qMLP as the current best simple candidate and seed-replicate it before H100/H200, width/depth, or stronger-stack integration.

### Phase 10: sp8192 Seed Replication

Goal: verify the stronger `sp8192` result is not a seed-42 outlier.

Use the exact Phase 9 benchmark shape:

```text
QUAT_MLP=1
QUAT_MLP_IMPL=matrix
VOCAB_SIZE=8192
DATA_PATH=/nfs/hpc/share/peterj29/pg/data-exports/sp8192-80/datasets/fineweb10B_sp8192
TOKENIZER_PATH=/nfs/hpc/share/peterj29/pg/data-exports/sp8192-80/tokenizers/fineweb_8192_bpe.model
TRAIN_BATCH_TOKENS=524288
VAL_BATCH_SIZE=524288
MAX_WALLCLOCK_SECONDS=600
```

Run at least two additional A40 seeds one at a time, matching the Phase 8 seed-replication discipline.

Decision gate:

- If both added seeds beat the `sp4096` mean and dense `sp1024`, promote `sp8192` to H100/H200 confirmation.
- If one seed regresses near or above the `sp4096` mean, run one more seed before deciding.
- If multiple seeds regress, keep `sp4096` as the safer simple candidate and defer H100/H200.
- Track the total int8+zlib submission size for every seed; stop if trained artifacts approach 16 MB.

### Phase 11: H100/H200 Confirmation

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

### Phase 12: Integration Into Strong Record Stack

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
