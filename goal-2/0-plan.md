# Lean A40 Plan for qMLP Parameter Golf Experiments

Date drafted: 2026-06-22

## Objective

Build a single-GPU A40 screening implementation from the best A40-friendly Parameter Golf ideas, then test whether matrix qMLP improves BPB across CaseOps vocabulary sizes.

The primary question is:

```text
Under a lean single-A40 screening setup, does qMLP improve best-under-16MB post-quant no-TTT BPB, and which CaseOps vocab size should be escalated to H100/FA3?
```

This plan intentionally avoids full phased TTT on A40. Full TTT is legal when score-first and within eval budget, but it is a fat evaluation method and was too slow on A40 to be a good exploration benchmark.

## Scope

In scope:

- single-GPU A40 training/eval/package runs;
- CaseOps tokenizer/data preparation;
- vocab sizes `1024`, `2048`, `4096`, `8192`, and `16384`;
- dense baseline variants at all listed vocab sizes as their CaseOps exports
  become available;
- qMLP variants at all listed vocab sizes;
- lean record-track ideas that transfer reasonably to A40:
  - CaseOps and original-byte BPB sidecars;
  - SP8192-style vocab defaults;
  - qMLP;
  - XSA;
  - SmearGate;
  - BigramHash or n-gram features if already local and cheap;
  - sparse attention gates / attention output gates;
  - partial RoPE and QK gain;
  - GPTQ/int6-int7 package checks;
  - SDPA/eager compatibility path.

Out of the A40 loop unless explicitly approved:

- full phased TTT benchmarks;
- H100/FA3 performance paths;
- 8xH100 reproduction;
- broad full-wallclock hyperparameter sweeps;
- lrzip/per-group compression if it dominates iteration time.

## Why SP8192 First

The mature record-track entries repeatedly converged on `sp8192`, especially with CaseOps. It is the best default baseline point for an A40-friendly record-style stack.

For this goal:

- `sp8192` remains the first sanity point because it is already available and is
  the common optimized record-track vocab;
- as the other agent finishes missing CaseOps exports, dense and qMLP are tested
  as paired runs at every ready vocab size;
- dense lower-vocab and higher-vocab results are no longer deferred controls;
  they are part of the baseline matrix needed to tell whether qMLP helps beyond
  vocab/package effects.

## CaseOps Vocabulary Policy

CaseOps vocab is not the same as the simple-stack SentencePiece vocab.

For each `VOCAB_SIZE`, prepare a CaseOps tokenizer by:

1. reading the selected FineWeb docs;
2. applying `encode_lossless_caps_v2`;
3. training SentencePiece BPE with CaseOps reserved control symbols;
4. using byte fallback, split digits, no dummy prefix, and reserved IDs;
5. exporting train shards, validation shards, and validation-byte sidecars with `prepare_caseops_data.py`;
6. scoring BPB on original bytes using the byte sidecars.

Dense and qMLP runs with the same `VOCAB_SIZE` must use the exact same CaseOps tokenizer model, vocab file, train shards, validation shards, and byte sidecars. The only intended difference between a paired dense/qMLP run is the model architecture flag.

Different vocab sizes require different CaseOps tokenizer/data exports:

```text
caseops-sp1024
caseops-sp2048
caseops-sp4096
caseops-sp8192
caseops-sp16384
```

## Benchmark Contract

Default A40 benchmark metrics:

- pre-quant `val_bpb`;
- post-quant no-TTT `val_bpb`;
- total submission bytes;
- quantized model bytes;
- training steps;
- ms/step;
- peak GPU memory;
- host/GPU;
- seed;
- command/config;
- run path and job ID.

Do not use final phased TTT BPB as the primary A40 score. If TTT is touched at all, use a tiny legality smoke only.

## OSU/HPC Execution Rules

These rules carry forward the server guidelines from `goal-1/0-plan.md` and the
repo-local `osu-hpc-agent-guide`. They govern every phase of this goal unless
the user explicitly changes them.

### Access Path And Host Roles

Use the OSU systems in this order:

1. Local Mac for editing, code review, planning, and git commits.
2. OSU Engineering gateway/flip server for access into the Engineering network.
3. OSU HPC submit node for Slurm inspection, job preparation, and submission.
4. Slurm compute nodes for any training, tokenizer/data export, compilation,
   GPU check, benchmark, or other material compute.

Observed shortcut path:

```bash
# Local Mac
osu

# OSU Engineering gateway / flip shell
hpc
```

Known alias findings:

```text
local:   osu='ssh peterj29@access.engr.oregonstate.edu'
gateway: hpc='ssh peterj29@submit-a.hpc.engr.oregonstate.edu'
```

For noninteractive inspection from the Mac, the equivalent path is nested SSH
through the gateway to `submit-a.hpc.engr.oregonstate.edu`. Use that only for
lightweight file inspection, Slurm control, and transfers. Do not run material
compute through nested SSH on a submit node.

The prior goal found `codex` on both the Engineering gateway and the HPC submit
node at `/nfs/stak/users/peterj29/.local/bin/codex`; verify this before relying
on a long remote Codex session.

### Authority And Live Checks

The live scheduler and current OSU policy are authoritative. Public docs and
old goal notes are planning references only.

Before submitting meaningful work, refresh live state from the submit node:

```bash
hostname -f
date --iso-8601=seconds
sinfo -a -o '%20P|%10a|%12l|%8D|%24F|%30G'
sinfo -a -N -o '%24N|%18P|%12T|%8c|%12m|%45G|%120f'
squeue -u "$USER"
sacctmgr -nP show assoc where user="$USER" \
  format=Cluster,Account,User,Partition,DefaultQOS,QOS 2>/dev/null || true
```

For A40 scheduler fit without submitting, use `srun --test-only` with an
explicit GPU request:

```bash
srun --test-only -p share --constraint=a40 --gres=gpu:1 \
  --time=00:10:00 --cpus-per-task=4 --mem=16G true

srun --test-only -p ampere --constraint=a40 --gres=gpu:1 \
  --time=00:10:00 --cpus-per-task=4 --mem=16G true

srun --test-only -p preempt --constraint=a40 --gres=gpu:1 \
  --time=00:10:00 --cpus-per-task=4 --mem=16G true
```

A feature constraint does not allocate a GPU. Every GPU job must include
`--gres=gpu:1` or the live supported equivalent.

### Submit-Node Boundary

Allowed on submit nodes:

- edit and inspect source;
- run Git commands;
- inspect Slurm, modules, quotas, and small text files;
- create batch scripts;
- submit, inspect, and cancel specifically identified jobs;
- run lightweight static checks such as `bash -n` and small parser checks.

Use Slurm compute allocations for:

- tokenizer/data export;
- training, eval, quantization, or packaging;
- GPU diagnostics including `nvidia-smi`;
- sustained CPU work;
- compilation over a material project;
- tests that use significant CPU, RAM, GPU, I/O, or walltime;
- any operation that opens or creates many files.

No training, tokenizer export, GPU diagnostics, or material compute should run
directly on submit nodes.

### Resource Guardrails

Default development envelope:

```text
nodes: 1
gpus: 1
cpus-per-task: 4-8 for GPU jobs
cpus-per-task: 16+ for full CaseOps CPU exports
memory: 16G-64G for smokes, measured larger only when needed
walltime: 10-30 minutes for smokes, bounded benchmark-specific time otherwise
```

Ask before requesting any of:

- H100/H200 resources;
- more than one GPU in a single job;
- more than 16 CPUs;
- more than 128G RAM;
- more than 8 hours;
- multi-node jobs;
- broad sweeps;
- destructive cleanup;
- cancellation of anything other than specifically identified jobs.

Parallelize independent A40 work when scheduler/account limits allow. For the
dense/qMLP vocab matrix, multiple concurrent one-A40 jobs are expected, with one
job per `(model_variant, vocab_size, seed)` cell. The required benchmark matrix
is three seeds per available vocab size for each core model: dense A40-friendly
CaseOps and qMLP. If queue pressure or QOS limits block full parallelism, run the
maximum schedulable subset and document the fallback.

Keep the dependency order:

```text
data export -> package/path smoke -> benchmark -> documentation update
```

Use `sbatch`, not long attached `srun`, for unattended work.

### Partition Guidance

Use live checks before each submission. Historical goal-1 findings were:

- `share/a40` was often the earliest accessible A40 path;
- `ampere/a40` can be useful for stable A40 work, but may preempt lower-priority
  work;
- `preempt/a40` is useful only for short, restartable, cheap work;
- H100/H200 should be held for explicit final confirmation after A40 evidence
  justifies it.

Do not hard-code these as permanent facts. The current Slurm state decides.

### Storage And Environment Rules

Use:

- `$HOME` only for shell config, small source, and small logs;
- `/nfs/hpc/share/peterj29/pg` for repo clones, environments, datasets, and run
  records;
- `/scratch/$USER/$SLURM_JOB_ID` for hot temporary data inside jobs.

Recommended remote layout:

```text
/nfs/hpc/share/peterj29/pg/
  src/pg/
  envs/
  data-exports/
  datasets/
  runs/
  wheels/
```

Do not put Python environments, FineWeb/CaseOps shards, model artifacts, or large
logs in home. The default `python3` on submit was previously observed as too old
for this project, so jobs should explicitly load a Python module or use a
controlled venv under `hpc-share`.

Stage heavy read/write workloads to local `/scratch` where practical, then copy
validated outputs back to `hpc-share`. Never delete the only copy of an input,
checkpoint, environment, or result.

### Job Records And Completion

Every meaningful job should preserve:

- Slurm job ID, partition, host, GPU, CPU/memory allocation, state, exit code,
  and elapsed time;
- exact command/config/env flags;
- git revision and short git status;
- module list and Python version;
- `nvidia-smi` output for GPU jobs;
- run path, logs, metrics, package sizes, and validation outputs.

A run is not successful merely because it left `squeue`. Completion requires:

- Slurm state is `COMPLETED`;
- exit code is `0:0`;
- expected outputs exist and are nonempty;
- validation/metrics parsing passes;
- stage-out completed;
- docs for the active phase are updated with the facts on the ground.

Explicitly report and handle `FAILED`, `OUT_OF_MEMORY`, `TIMEOUT`, `PREEMPTED`,
`CANCELLED`, or `NODE_FAIL`.

## Phase Plan

### Phase 0: Inventory

Goal: capture the current repo, remote state, existing CaseOps exports, and usable Slurm partitions without starting new training.

Actions:

1. Check local and remote git status.
2. List existing CaseOps exports for `1024`, `2048`, `4096`, `8192`, and `16384`.
3. List existing qMLP patches and record-stack patches.
4. Check user queue and live A40 availability.
5. Record whether old `goal-1` jobs are still running, but do not cancel them without approval.

Exit criteria:

- inventory phase file exists;
- no compute work has run on submit nodes;
- next data/export phase knows what can be reused.

### Phase 1: CaseOps Data Matrix

Goal: prepare or verify CaseOps tokenizer/data exports for the vocab ladder.

Actions:

1. Reuse completed CaseOps exports when valid.
2. For missing vocab sizes, run CPU Slurm exports with 16+ CPUs.
3. Use bounded smoke exports before full 80-shard exports when a tokenizer path or script path is uncertain.
4. Verify tokenizer model, vocab file, manifest, train shards, validation shards, and byte sidecars.
5. Confirm that paired dense/qMLP runs at each vocab size will point to the same data paths.

Vocab targets:

```text
1024
2048
4096
8192
16384
```

Exit criteria:

- every target vocab is either ready or has a captured blocker;
- `sp8192` is ready before matrix benchmarking starts;
- data paths are recorded in the phase file.

### Phase 2: Lean A40 Harness

Goal: create a reusable single-A40 harness for the lean record-style stack.

Default lean settings:

```text
TTT_ENABLED=0
PHASED_TTT_ENABLED=0
DOCUMENT_PACKING=0 unless proven safe
TORCH_COMPILE=0 unless proven safe
FUSED_MLP_ENABLED=0 unless proven safe
FUSED_CE_ENABLED=0 unless proven safe
attention=SDPA/eager fallback
single GPU
post-quant no-TTT BPB is primary
```

Record-style lean features to include if already local and cheap:

- CaseOps;
- XSA;
- SmearGate;
- sparse attention gate / attention output gate;
- partial RoPE;
- QK gain;
- quantization/package path.

Exit criteria:

- one script or small set of scripts can run dense or qMLP by env flags;
- script records all benchmark contract metrics;
- smoke run reaches package output on A40.

### Phase 3: Parallel Package Smoke Matrix

Goal: verify package/path viability for dense and qMLP at every available CaseOps
vocab size before spending full A40 benchmark cycles.

Matrix:

```text
model variants: dense, qmlp
vocab sizes:    1024, 2048, 4096, 8192, 16384
seed:           42 for smoke
```

Actions:

1. Build a live matrix from the Phase 1 data inventory. Include a cell only when
   its CaseOps tokenizer, train shards, validation shards, and byte sidecars are
   verified.
2. Submit independent one-A40 smoke/package jobs in parallel for every ready
   `(model_variant, vocab_size)` cell.
3. Use Slurm dependencies for cells whose data export is still running, so the
   smoke starts only after the matching data job succeeds.
4. Record job ID, host, Slurm state, exit code, package size, pre/post-quant BPB,
   peak memory, and exact data/tokenizer path.
5. Do not benchmark cells whose smoke fails, exceeds 16 MB, or points at stale or
   mismatched data.

Exit criteria:

- every currently available vocab has dense and qMLP smoke outcomes, or a
  captured blocker;
- newly completed vocab exports can be added as follow-up matrix cells without
  rewriting the harness;
- at least `sp8192` dense and `sp8192` qMLP have complete package-smoke results.

### Phase 4: Parallel Three-Seed A40 Matrix

Goal: get three-seed A40 measurements for both core models across all available
CaseOps vocab sizes.

Matrix:

```text
model variants: dense, qmlp
vocab sizes:    every verified CaseOps vocab from Phase 1
seeds:          42, 0, 1
score:          post-quant no-TTT BPB
```

Actions:

1. For every Phase 3 passing `(model_variant, vocab_size)` smoke cell, submit
   seeds `42`, `0`, and `1` as independent one-A40 benchmark jobs.
2. Run cells in parallel where Slurm/account limits allow. Prefer job arrays or a
   manifest-driven launcher with a concurrency cap over hand-submitting many
   one-off commands.
3. Keep dense and qMLP paired at each vocab size and seed: same data path, same
   seed, same lean settings, only the architecture flag changes.
4. As the other agent finishes missing vocab exports, append those cells to the
   same matrix rather than waiting for the entire suite before using idle A40s.
5. Record pre-quant BPB, post-quant no-TTT BPB, total submission bytes, quantized
   model bytes, train steps, ms/step or tok/s, peak memory, host/GPU, command,
   job ID, and run path.

Exit criteria:

- every available CaseOps vocab has dense and qMLP benchmark outcomes for seeds
  `42`, `0`, and `1`, or a documented blocker;
- all completed benchmark cells are under 16 MB or explicitly marked over budget;
- the matrix is documented in the active phase file with enough metadata to
  reproduce each cell.

### Phase 5: Matrix QA And Reruns

Goal: repair missing or failed cells and summarize seed stability after the
required three-seed matrix.

Actions:

1. Inspect Phase 4 for failed, timed-out, over-budget, or missing matrix cells.
2. Rerun only cells where the failure appears transient or where a paired dense
   versus qMLP comparison would otherwise be incomplete.
3. Do not add new seeds by default; the required seed set is `42`, `0`, and `1`.
4. Compute per-vocab/per-model means, standard deviations, and paired seed
   deltas.
5. Record any host/GPU differences that could explain outlier timing or BPB.

Exit criteria:

- each valid `(model_variant, vocab_size)` cell has three seeds or an explicit
  blocker;
- qMLP-vs-dense comparisons are paired by vocab and seed whenever possible;
- incomplete cells are labeled exploratory and excluded from mean-based decisions
  unless the missingness is justified.

### Phase 6: Matrix Analysis And Budget Controls

Goal: determine whether qMLP helps because of architecture, budget reinvestment,
or neither.

Actions:

1. Compare dense versus qMLP at each vocab size.
2. Compare within-variant vocab trends for dense and qMLP.
3. Identify whether qMLP's package savings buy a better vocab point under the
   16 MB cap.
4. Use additional dense budget controls only if the matrix leaves the conclusion
   ambiguous.
5. Do not extend beyond `sp16384` unless the user explicitly redirects.

Exit criteria:

- matrix summary exists for dense and qMLP across available vocab sizes;
- best-under-16MB dense and best-under-16MB qMLP candidates are identified;
- the analysis says whether more A40 work is warranted before H100/FA3.

### Phase 7: A40 Decision

Goal: decide what, if anything, deserves H100/FA3 confirmation.

Actions:

1. Summarize the dense and qMLP vocab matrix.
2. Separate lean gains from fat dependencies.
3. Identify whether the candidate should be ported to H100/FA3 with full legal eval.
4. Update all `goal-2` docs.

Exit criteria:

- keep/modify/abandon qMLP decision exists;
- H100 candidate command is drafted only if A40 evidence justifies it;
- no H100/H200 work is launched without explicit approval.

## Current Starting Assumptions

- `goal-1` has been renamed and is no longer the active loop.
- qMLP code and CaseOps export scripts exist from prior work, but must be re-inventoried.
- CaseOps `sp8192` and `sp16384` exports may already exist remotely.
- another Codex agent may be creating the missing CaseOps `sp1024`, `sp2048`,
  and `sp4096` exports; add those vocab cells to the benchmark matrix only after
  their data artifacts verify cleanly.
- `sp8192` is the most common optimized record-track vocab and should be the
  first sanity point, not the only dense baseline.
- Full A40 phased TTT is too slow to be a useful default benchmark.

## Success Criteria

The goal succeeds if it produces:

- a reusable lean A40 benchmark harness;
- a paired dense/qMLP CaseOps vocab matrix across all available target vocabs;
- parallel A40 smoke records and three-seed benchmark records for each valid
  matrix cell;
- a defensible best dense candidate and best qMLP candidate under 16 MB;
- a defensible decision to keep, modify, or abandon qMLP;
- clear docs separating A40 screening evidence from record-compliant evidence.
