# Goal 3 Plan: 8xH100 qMLP Record Attempt

Date drafted: 2026-06-23

## Objective

Prepare and execute a full 8xH100 Parameter Golf record-track attempt that starts
from the strongest known compliant competition stack and adds qMLP in the most
controlled way possible.

The decision question is:

```text
Can qMLP improve the best-under-16MB record-track result on the intended
8xH100 H100/FA3 competition setup?
```

This is not an A40 exploration goal. A40 evidence has already done its job:
Goal 2 showed that qMLP is worth carrying forward. Goal 3 is about making the
H100 allocation answer the record question cleanly.

## Current Facts To Carry Forward

- The competition target is a 16 MB artifact cap and a 10-minute training budget
  on 8xH100.
- The strongest local H100 record candidate is:

```text
parameter-golf/records/track_10min_16mb/
2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611
```

- That record reports:
  - 3-seed post-TTT mean `1.06108` BPB;
  - about `15.9 MB` artifacts;
  - about `4931` training steps in `600s`;
  - about `121.7 ms/step`;
  - 8xH100 SXM 80GB;
  - FA3, CaseOps `sp8192`, XSA, sparse attention gate, SmearGate BOS fix,
    Polar-Express Muon, fused CE, LQER, GPTQ, per-group `lrzip` compression,
    and phased score-first TTT.
- A closely related compliance reproduction is:

```text
parameter-golf/records/track_10min_16mb/
2026-04-29_SmearGateBOSFix_3Seed_1.06141
```

- Goal 2 found strong A40-screening evidence for qMLP in an A40-friendly
  CaseOps record-track harness:
  - best clean dense control: `sp8192`, mean quantized BPB `3.66039918`;
  - best clean qMLP candidate: `sp16384`, mean quantized BPB `2.99609830`;
  - qMLP beat dense at every matched vocab size tested;
  - qMLP `sp32768` remained under cap but was worse than `sp16384`;
  - the dominant Goal 2 gain was matched-vocab qMLP, with vocab reinvestment
    adding a smaller additional benefit.
- Current live OSU Slurm information is not permanent. Previous live checks
  showed the `dgxh` partition could accept an 8-GPU H100 request and the user
  QOS allowed up to 8 GPUs and up to 2880 GPU-minutes active, but every H100
  request must re-check live state before submission.
- As of the 2026-06-23 Phase 0 live check, `--constraint=h100` alone was too
  broad: Slurm dry-run targeted `dgxh-1`, which is advertised as
  `gpu:h100-40g:16`. Use an 80GB constraint for the intended competition-class
  node, currently `--constraint="h100&vram80g"`, and re-check this live before
  submission.
- `codex` has been observed on the OSU Engineering gateway and HPC submit node
  at `/nfs/stak/users/peterj29/.local/bin/codex`, but this must be reverified
  before any Slurm job relies on it.

## Strategic Position

The H100 node should be treated as an execution window, not an exploration
window. All code, configs, data, packaging, Slurm scripts, parsers, and fallback
logic should be prepared before requesting the node.

Revised campaign requirement: do not queue a tiny env smoke separately from the
real run. H100 wait time is the scarce resource. The next H100 request should be
one autonomous campaign allocation that validates the environment, catches build
issues, reproduces the known record-setting dense baseline closely enough to
trust the setup, and then runs the qMLP contender for three seeds.

The preferred autonomous campaign sequence is:

1. Runtime setup/build validation:
   - activate the prepared env;
   - attempt FA3 install/build from the documented wheel source if the import is
     missing and runtime install is enabled;
   - verify CUDA/PyTorch, 8 H100 80GB GPUs, FA3, Triton, SentencePiece, Brotli,
     `lrzip`, and the two CaseOps tokenizers.
2. Exact dense/base `sp8192` full baseline run, seed 42.
3. Stop only if the baseline is not credible:
   - candidate exits nonzero;
   - artifact is not under 16 MB;
   - post-TTT BPB is worse than the configured hard ceiling;
   - training step count is below the configured hard floor.
   A stricter parity target is still recorded, but missing it should caveat the
   comparison rather than automatically waste the allocation.
4. qMLP `sp16384` full contender, seeds `42`, `0`, and `1234`.
5. Save all logs, full model artifacts, quantized submission artifacts,
   per-candidate status, hashes, summaries, and final campaign summary.

Do not spend scarce H100 time on open-ended debugging, broad sweeps, or
interactive agent exploration.

## Base Stack And Candidate Set

### Primary Base

Use the 2026-04-27 record candidate as the primary base unless implementation
inspection finds a concrete blocker:

```text
2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611
```

Reason:

- It is the strongest local H100 record note.
- It already contains the full H100/FA3, CaseOps, TTT, GPTQ/LQER, and
  compression stack we ultimately need to beat.
- It provides exact reported seed results, commands, requirements, and artifact
  size context.

### Fallback Base

Use the 2026-04-29 compliance reproduction only if the 2026-04-27 stack proves
too hard to stage or legally reproduce within the OSU environment:

```text
2026-04-29_SmearGateBOSFix_3Seed_1.06141
```

Reason:

- It has clearer compliance timing notes around `GPTQ_RESERVE_SECONDS=8.0`.
- It is slightly weaker but more explicitly packaged as a compliance
  reproduction.

### Required Candidate Runs

The minimum autonomous H100 campaign must prepare these configs:

| Priority | Config | Purpose |
|---:|---|---|
| 1 | exact dense/base `sp8192`, seed 42, full run | prove OSU H100 setup can reproduce the known record path closely enough |
| 2 | qMLP `sp16384`, seed 42, full run | first contender result after baseline hard validity passes |
| 3 | qMLP `sp16384`, seed 0, full run | seed replication |
| 4 | qMLP `sp16384`, seed 1234, full run | seed replication and direct comparison to record seed set |

Optional configs, only if already scripted and time remains:

| Priority | Config | Purpose |
|---:|---|---|
| 5 | qMLP same-vocab `sp8192` smoke or full run | isolate qMLP same-vocab effect if campaign headroom remains |
| 6 | qMLP `sp32768` package smoke only | check whether the H100/full-stack package frontier differs materially |

Do not prioritize `sp32768` for full H100 training unless `sp16384` is already
successful and package/throughput evidence says `sp32768` is likely to improve.
Goal 2 showed `sp32768` was under cap but worse than `sp16384`.

## Compliance Rubric

Every final candidate must satisfy:

- artifact under `16,000,000` bytes using the same submission accounting used by
  the record scripts;
- no training-data access beyond the allowed training budget;
- score-first TTT only, if TTT is used;
- no cross-document leakage;
- CaseOps original-byte BPB sidecar accounting preserved;
- exact data/tokenizer paths recorded;
- exact command, seed, host, GPUs, Slurm job ID, Git SHA, and dirty diff
  recorded;
- no submit-node training, GPU diagnostics, data export, compilation, or
  material compute;
- all H100/H200 requests explicitly approved by the user before submission.

If compliance is uncertain, stop and document the uncertainty before launching
more scarce work.

## OSU/HPC Operating Rules

Follow `osu-hpc-agent-guide` and the carried-forward Goal 1/Goal 2 rules.

Access path:

```text
local Mac -> OSU Engineering gateway/flip -> HPC submit node -> Slurm compute node
```

Known aliases:

```bash
osu='ssh peterj29@access.engr.oregonstate.edu'
hpc='ssh peterj29@submit-a.hpc.engr.oregonstate.edu'
```

Submit nodes are control planes only. They may be used for:

- editing and file inspection;
- Git;
- Slurm inspection and job submission;
- small text parsing;
- `bash -n` and lightweight static checks.

Use Slurm allocations for:

- H100/A40 GPU checks;
- training/eval/quantization/packaging;
- tokenizer/data export;
- FA3 or Triton compilation;
- sustained CPU or I/O work.

Before meaningful submissions, refresh live state:

```bash
hostname -f
date --iso-8601=seconds
squeue -u "$USER"
sinfo -a -o '%20P|%10a|%12l|%8D|%24F|%30G'
sinfo -a -N -o '%24N|%18P|%12T|%8c|%12m|%45G|%120f'
sacctmgr -nP show assoc where user="$USER" \
  format=Cluster,Account,User,Partition,DefaultQOS,QOS 2>/dev/null || true
sacctmgr -nP show qos format=Name,MaxTRESPU,MaxTRESRunMinsPU 2>/dev/null || true
```

Before requesting 8xH100, dry-run:

```bash
srun --test-only -p dgxh --constraint="h100&vram80g" --gres=gpu:8 \
  --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=500G \
  --time=06:00:00 true
```

The current campaign target is six hours because it must include environment
validation, one full dense baseline parity run, and three full qMLP seed runs in
one queue wait without failing from tight packaging/TTT/eval timeouts. This
matches the observed `dgxh` QOS GPU-minute ceiling for one 8-GPU job.

## Required Artifacts To Prepare Before H100 Request

The following must exist, be reviewed, and pass non-H100 checks before asking
for the 8xH100 allocation.

### Source Artifacts

- A clean working branch for Goal 3.
- A chosen base record copied or wrapped in a Goal 3 staging area.
- qMLP implementation ported into the selected record stack with minimal
  unrelated changes.
- Config flags that allow switching between:
  - dense/base MLP;
  - qMLP same-vocab;
  - qMLP budget-reinvested vocab.
- Staged flag policy:
  - `QUAT_MLP=0` for dense/base;
  - `QUAT_MLP=1` for qMLP.
- A diff summary showing exactly what changed from the base record.
- A compliance note explaining why the qMLP change does not change data access
  or evaluation legality.
  Current file: `goal-3/compliance-note.md`.

### Data And Tokenizer Artifacts

- Exact CaseOps `sp8192` tokenizer and data path for the selected base stack.
- CaseOps `sp16384` tokenizer and data path compatible with the selected stack.
- Validation byte sidecars verified for both vocab sizes.
- A small manifest for each data export:
  - tokenizer path;
  - vocab size;
  - train shard count;
  - validation shard count;
  - byte sidecar path;
  - generating script and Git SHA;
  - creation job ID if produced on HPC.

Dense/base and qMLP runs at the same vocab must use the exact same tokenizer,
shards, and sidecars.

### Environment Artifacts

- Versioned Python environment path under `/nfs/hpc/share/peterj29`, not `$HOME`.
- Captured requirements for the selected record stack.
- PyTorch/CUDA target documented.
- FA3 install method documented.
- `lrzip` availability checked or an approved install/load path documented.
  Current user-local prep script: `goal-3/prepare-tools.sbatch`, targeting
  `/nfs/hpc/share/peterj29/pg/tools/lrzip`.
- Triton/fused-kernel compile behavior tested in a non-scarce allocation where
  possible.
- Environment smoke script that records:
  - `python -VV`;
  - `pip freeze` or equivalent;
  - `module list`;
  - `torch.__version__`;
  - CUDA availability;
  - FA3 import status;
  - `lrzip --version`.

### Slurm Scripts

Create and review these scripts before H100 submission:

```text
goal-3/prepare-env.sbatch
goal-3/prepare-tools.sbatch
goal-3/h100-env-smoke.sbatch
goal-3/h100-short-smoke.sbatch
goal-3/h100-record-runner.sbatch
goal-3/h100-campaign-runner.sbatch
goal-3/h100-repair-agent.sbatch
goal-3/scripts/env_smoke.py
goal-3/scripts/run_candidate.sh
goal-3/scripts/parse_train_log.py
goal-3/scripts/static_goal3_audit.py
```

The final runner must:

- request exactly one H100 80GB node and 8 H100 80GB GPUs;
- explicitly set partition, constraint, GRES, time, nodes, tasks, CPUs, memory,
  stdout, and stderr;
- record host, Slurm env, Git SHA, dirty status, modules, GPU inventory, and
  dependency versions;
- create a unique run directory using `$SLURM_JOB_ID`;
- stage hot inputs to local scratch if practical;
- run hardware/env smoke first;
- validate/build missing FA3 runtime pieces if possible before failing;
- run full dense/base baseline before qMLP attempts;
- execute qMLP `sp16384` for three predeclared seeds after the baseline hard
  validity gate passes;
- stop on invalid package size or compliance failure;
- preserve full model artifacts, quantized submission artifacts, logs,
  manifests, hashes, a Goal 3 source snapshot, parser output, and final
  campaign summaries on shared storage before exit;
- write a final machine-readable status file.

The repair-agent script must be optional and disabled by default. If used, it
must run Codex with:

- `codex exec`, not an interactive TUI;
- a strict prompt from `goal-3/0-prompt.md`;
- `timeout`;
- bounded workspace-write access;
- no broad destructive commands;
- no new H100 submissions;
- no broad sweeps;
- logs written into the current run directory.

### Parser And Reporting Artifacts

Create parsers or shell summaries that extract:

- Slurm job ID, state, exit code, elapsed;
- host and GPU inventory;
- seed;
- candidate config;
- train steps;
- ms/step or tokens/s;
- pre-quant BPB;
- post-quant BPB;
- post-TTT BPB when applicable;
- artifact bytes;
- package status under/over 16 MB;
- training-data-access timing;
- TTT/eval timing;
- path to final submission artifact.

Required files:

```text
goal-3/status.md
goal-3/jobs.csv
goal-3/findings-summary.md
```

These may be created during implementation phases, but they must exist before
the final H100 request is submitted.

## Phases

### Phase 0: Inventory And Source Of Truth

Goal: establish the Goal 3 working state without starting H100 work.

Steps:

1. Read `goal-3/0-plan.md`, `goal-3/0-loop.md`, and `goal-3/0-prompt.md`.
2. Read `goal-2/findings-summary.md`.
3. Read the selected base record README and `train_gpt.py`.
4. Read the OSU HPC guide and refresh live Slurm state.
5. Record current local Git branch, Git SHA, dirty files, submodule SHA, and
   remote HPC source path.
6. Create or refresh `goal-3/status.md` and `goal-3/jobs.csv`.

Completion requirements:

- no H100/H200 jobs submitted;
- live scheduler/account facts recorded with timestamp;
- selected base record and fallback base recorded;
- current next action is clear in `goal-3/status.md`.

### Phase 1: Base Record Selection And Compliance Map

Goal: decide exactly which record stack is the base and document its compliance
surface before modifying it.

Steps:

1. Compare the 2026-04-27 and 2026-04-29 record stacks.
2. Identify required dependencies, data layout, H100 assumptions, and packaging
   tools.
3. Identify where the MLP block, quantization, package accounting, and TTT hooks
   live.
4. Write `goal-3/1-base.md` with:
   - selected base;
   - fallback base;
   - record command;
   - required env vars;
   - compliance-sensitive code paths;
   - exact places qMLP may touch.

Completion requirements:

- `goal-3/1-base.md` exists;
- selected base is justified;
- qMLP insertion points are identified;
- compliance risks are listed before implementation.

### Phase 2: Data And Tokenizer Readiness

Goal: ensure all required CaseOps data exists and matches the selected stack.

Steps:

1. Verify the base `sp8192` CaseOps tokenizer and data path.
2. Verify or create compatible `sp16384` CaseOps tokenizer/data.
3. Confirm original-byte sidecars exist and match validation shards.
4. Run only lightweight validation on submit nodes; use CPU Slurm jobs for any
   export or heavy verification.
5. Write manifests for each vocab.

Completion requirements:

- `sp8192` and `sp16384` manifests exist;
- dense/qMLP same-vocab pairing policy is explicit;
- no data export ran on a submit node;
- missing data is either produced or documented as a blocker.

### Phase 3: qMLP Port Into Full Record Stack

Goal: implement qMLP in the selected H100 record stack with the smallest useful
change.

Steps:

1. Port the matrix qMLP implementation into the selected `train_gpt.py` stack.
2. Add the `QUAT_MLP=1` feature flag and keep `QUAT_MLP=0` as the dense/base
   path.
3. Keep the dense/base path byte-for-byte or behaviorally unchanged when qMLP is
   disabled.
4. Ensure qMLP participates correctly in:
   - optimizer parameter grouping;
   - Muon or Adam selection;
   - EMA;
   - quantization;
   - LQER/GPTQ hooks;
   - serialization and compression;
   - TTT LoRA hooks, if applicable.
5. Add a CPU or tiny-GPU forward/shape smoke where possible.
6. Write `goal-3/3-qmlp-port.md` documenting the diff.

Completion requirements:

- dense/base path still runs its import/config smoke;
- qMLP path runs a tiny forward/shape smoke;
- package code can see qMLP tensors;
- all known qMLP-related diffs are documented.

### Phase 4: Package And Artifact Smokes

Goal: script the cheapest package and artifact checks, and run them before any
full record candidate inside an approved allocation. Do not pretend package
size is proven until the staged qMLP code actually serializes a candidate.

Steps:

1. Script package/runtime smokes for:
   - base/dense `sp8192`;
   - qMLP `sp8192`;
   - qMLP `sp16384`.
2. Use cheap smoke data and minimal training where possible.
3. Confirm final accounting uses total submission bytes, not only model bytes.
4. Check whether qMLP `sp16384` has enough headroom for the full record stack.
5. Stop `sp32768` unless `sp16384` has large headroom and there is a specific
   reason to inspect it.
6. If FA3/H100-only dependencies prevent an honest non-H100 package smoke, run
   the smoke first inside the approved H100 allocation and stop before the full
   contender if package accounting or qMLP runtime is invalid.

Completion requirements:

- package smoke scripts exist;
- package smoke results recorded once the approved runtime smoke runs;
- under/over cap status known for required candidates before treating any full
  result as valid;
- any over-budget candidate is removed from final-run priority.

### Phase 5: Non-Scarce Runtime Smokes

Goal: catch ordinary bugs before touching H100.

Steps:

1. Run import/config smokes.
2. Run a tiny CPU or A40 qMLP smoke if compatible.
3. Run a one-GPU Slurm smoke on available non-H100 GPU if useful.
4. Verify logs and parsers work.
5. Avoid full A40 TTT unless explicitly useful; Goal 2 already showed A40 TTT
   is not the right exploration loop.

Completion requirements:

- qMLP candidate starts training and produces loss movement in a bounded smoke;
- output paths, manifests, and parsers work;
- no known syntax/import/path errors remain;
- failure modes are documented.

### Phase 6: H100 Runner Implementation

Goal: create the unattended runner that will use the 8xH100 allocation safely.

Steps:

1. Write `goal-3/h100-env-smoke.sbatch`.
2. Write `goal-3/h100-short-smoke.sbatch`.
3. Write `goal-3/h100-record-runner.sbatch`.
4. Write `goal-3/h100-campaign-runner.sbatch` as the primary approval target.
5. Write `goal-3/h100-repair-agent.sbatch`, disabled by default.
6. Add parser/reporting scripts.
7. Validate scripts with `bash -n`.
8. Use `srun --test-only` for scheduler fit checks.

Completion requirements:

- all required scripts exist and pass static checks;
- static Goal 3 audit verifies qMLP wiring and H100 runner guardrails;
- campaign runner has a baseline hard validity gate, strict parity note, three
  qMLP seeds, and no hidden broad sweep;
- final runner writes final status even on failure;
- repair agent is bounded, optional, and not the default execution path.

### Phase 7: Human Review And H100 Request Gate

Goal: obtain explicit approval before scarce H100 submission.

Before submission, present:

- exact Slurm command or `sbatch` script path;
- requested partition, constraint, GPUs, CPUs, memory, and walltime;
- current dry-run estimated start, if available;
- exact candidate run order;
- expected maximum H100 time consumed;
- stop conditions;
- artifact/log destination;
- known risks and fallback order.

Completion requirements:

- user explicitly approves the H100 request;
- current Slurm live state is recorded;
- final script path and Git SHA are recorded;
- no H100 submission happens without approval.

### Phase 8: H100 Execution Window

Goal: run the preplanned H100 sequence and capture enough evidence to decide
whether qMLP beats the record.

The campaign runner should execute:

1. Runtime setup/build validation:
   - activate env;
   - install/build FA3 from the documented wheel source if missing and enabled;
   - `nvidia-smi -L`;
   - confirm 8 full H100 GPUs;
   - record CUDA and NCCL state.
2. Environment verification:
   - Python;
   - PyTorch CUDA;
   - FA3 import;
   - Triton/fused kernels;
   - `lrzip`.
3. Full baseline parity:
   - dense/base `sp8192`, seed 42;
   - validate under-16MB artifact, hard BPB ceiling, and hard step-count floor;
   - record strict parity target pass/fail separately for interpretation.
4. Full qMLP campaign:
   - qMLP `sp16384`, seed 42;
   - qMLP `sp16384`, seed 0;
   - qMLP `sp16384`, seed 1234.
5. Stage-out:
   - logs;
   - full model artifacts;
   - quantized submission artifacts;
   - artifact hashes;
   - parser summaries;
   - final status.

Completion requirements:

- H100 job reaches terminal Slurm state;
- logs and artifacts are copied back to shared storage;
- parser summary exists;
- any failed step has enough logs for diagnosis;
- package compliance is known for every completed run;
- qMLP three-seed mean/std are computed if all three qMLP runs complete.

### Phase 9: Post-Run Analysis And Next Decision

Goal: decide whether Goal 3 succeeded, needs one more H100 run, or should stop.

Steps:

1. Parse all H100 outputs.
2. Compare against:
   - 2026-04-27 reported mean `1.06108`;
   - 2026-04-29 reported mean `1.06141`;
   - any exact dense/base reproduction run produced on OSU.
3. Separate:
   - same-vocab qMLP effect;
   - vocab reinvestment effect;
   - throughput loss or gain;
   - package-size effect;
   - TTT/quantization interactions.
4. Update `goal-3/findings-summary.md`.
5. If a candidate is record-breaking and compliant, write the submission package
   checklist.
6. If not, document the blocker and stop unless there is a specific cheap next
   run.

Completion requirements:

- `goal-3/findings-summary.md` exists and is current;
- success/failure is stated plainly;
- no active Slurm jobs remain untracked;
- next action is either an approved follow-up or stop.

## Stop Conditions

Stop and ask before continuing if:

- H100 package size is over 16 MB;
- qMLP same-vocab smoke is clearly broken inside the record stack;
- dense/base baseline parity fails badly enough that OSU setup is not credible;
- FA3 or fused kernels cannot run on OSU H100;
- the runner would need unplanned H100 debugging;
- compliance is ambiguous;
- the exact Slurm request differs materially from the reviewed request;
- a new idea would require broad H100 sweeps.

## Success Criteria

Minimum useful success:

- qMLP full-stack H100 candidate runs to completion;
- artifact is under 16 MB;
- BPB and timing are parsed;
- result is interpretable against the known record stack.

Strong success:

- qMLP `sp16384` beats the local reported `1.06108` H100 record candidate or
  produces a clear record-level result within seed noise that justifies another
  seed.

Full success:

- qMLP produces a compliant, under-16MB, full-stack 8xH100 result that beats the
  known local record by enough margin to justify treating it as a record-breaking
  submission candidate.
