# Prompt: Goal 3 8xH100 qMLP Record Attempt

You are continuing the work governed by `goal-3/0-plan.md` and
`goal-3/0-loop.md`.

Your objective is to prepare and execute a full 8xH100 Parameter Golf
record-track attempt that starts from the strongest known compliant competition
stack and adds qMLP in a controlled way.

The decision question is:

```text
Can qMLP improve the best-under-16MB record-track result on the intended
8xH100 H100/FA3 competition setup?
```

## First Actions

Before doing implementation work:

1. Read `goal-3/0-plan.md`.
2. Read `goal-3/0-loop.md`.
3. Read `goal-2/findings-summary.md`.
4. Read the relevant OSU HPC guide files, especially:
   - `osu-hpc-agent-guide/AGENTS.md`;
   - `osu-hpc-agent-guide/ssh-setup.md`;
   - `osu-hpc-agent-guide/OSU_HPC_AGENT_GUIDE.md` as needed.
5. Read the primary base record README:

```text
parameter-golf/records/track_10min_16mb/
2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/README.md
```

6. Read the fallback/compliance record README:

```text
parameter-golf/records/track_10min_16mb/
2026-04-29_SmearGateBOSFix_3Seed_1.06141/README.md
```

Then update or create `goal-3/status.md` with the current state and next action.

## Operating Rules

Follow `goal-3/0-loop.md`.

Facts on the ground beat stale plans. If code inspection, package sizes, Slurm
state, logs, or compliance findings contradict the written plan, update the docs
and follow the new facts.

Do not run training, tokenizer export, GPU diagnostics, compilation, or material
compute on submit nodes. Use Slurm compute allocations.

Do not submit H100/H200 jobs without explicit user approval after showing:

- exact script path;
- exact Slurm resources;
- requested walltime;
- candidate run order;
- stop conditions;
- artifact/log destination;
- known risks.

Do useful ready work in parallel where safe. Do not wait idly for one dependency
if another prerequisite can move. Do not bypass gates that protect H100 time.

## Campaign Shape

The H100 allocation should run prewritten scripts, not open-ended exploration.

Prepare these required artifacts before asking for H100:

```text
goal-3/status.md
goal-3/jobs.csv
goal-3/findings-summary.md
goal-3/compliance-note.md
goal-3/prepare-env.sbatch
goal-3/prepare-tools.sbatch
goal-3/h100-env-smoke.sbatch
goal-3/h100-short-smoke.sbatch
goal-3/h100-record-runner.sbatch
goal-3/h100-repair-agent.sbatch
goal-3/scripts/env_smoke.py
goal-3/scripts/run_candidate.sh
goal-3/scripts/parse_train_log.py
```

For a one-hour 8xH100 allocation, the default candidate order should prioritize
the record attempt after smoke validation:

```text
dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384
```

The required candidate configs are:

1. exact dense/base `sp8192` smoke;
2. qMLP same-vocab `sp8192` smoke or full run, depending on approved time;
3. qMLP budget-reinvested `sp16384` full contender.

Optional only if already scripted and time remains:

1. exact dense/base full `sp8192` reproduction;
2. qMLP `sp16384` second seed;
3. qMLP `sp32768` package smoke.

Do not prioritize `sp32768` full H100 training unless `sp16384` is already
successful and there is concrete package/throughput evidence that it is likely
to improve.

## Implementation Guidance

Use the 2026-04-27 record as the primary base unless inspection finds a concrete
blocker. Use the 2026-04-29 compliance reproduction as fallback.

Port qMLP into the selected record stack with minimal unrelated changes.

When qMLP is disabled, the dense/base path should remain behaviorally unchanged.
The staged implementation uses `QUAT_MLP=0` for dense/base and `QUAT_MLP=1` for
qMLP.

Make qMLP compatible with:

- optimizer parameter grouping;
- Muon or Adam assignment;
- EMA;
- quantization;
- LQER/GPTQ hooks;
- serialization and compression;
- TTT LoRA hooks, if the selected stack applies LoRA to MLP paths.

Dense/base and qMLP runs at the same vocab must use the same CaseOps tokenizer,
train shards, validation shards, and original-byte sidecars.

Preserve compliance-sensitive behavior:

- 16 MB artifact accounting;
- training-data-access timing;
- score-first TTT;
- document-boundary behavior;
- CaseOps original-byte BPB sidecar accounting.

## H100 Runner Requirements

The final H100 runner must:

- request one H100 80GB node and 8 H100 80GB GPUs;
- explicitly set partition, constraint, GRES, time, nodes, tasks, CPUs, memory,
  stdout, and stderr;
- record host, Slurm env, Git SHA, dirty status, modules, GPU inventory, and
  dependency versions;
- create a unique run directory using `$SLURM_JOB_ID`;
- stage hot inputs to local scratch if practical;
- run hardware/env smoke first;
- run a short distributed qMLP smoke before final attempts;
- execute only predeclared candidate configs;
- stop on invalid package size or compliance failure;
- copy logs, artifacts, manifests, parser summaries, and final status back to
  shared storage before exit.

The runner should record `git-status.txt`, `git-diff.stat`, and
`git-diff.patch`, and should stage the small Goal 3 source/data inputs to
`/scratch/$USER/$SLURM_JOB_ID/goal3` when practical.

Use `srun --test-only` before submitting real H100 work. Do not use
`--constraint=h100` alone: the Phase 0 live check showed that can target
`dgxh-1` with `gpu:h100-40g:16`. For the intended 8xH100 80GB class, use the
live-validated stricter constraint, currently `--constraint="h100&vram80g"`.

Do not claim package feasibility until the staged qMLP code has actually
serialized a candidate and emitted total submission bytes. If FA3/H100-only
dependencies prevent a non-H100 package smoke, the approved H100 runner must do
the smoke/package gate first and stop before the full contender if it fails.

## Codex-On-Node Policy

Codex exists on the OSU/HPC path, but do not rely on an open-ended Codex session
as the primary H100 execution method.

The default path is deterministic Slurm scripts.

`goal-3/h100-repair-agent.sbatch` may exist as an optional bounded fallback. If
used, it must call `codex exec` with a timeout and this prompt, write logs into
the current run directory, and must not submit new H100 jobs, launch broad
sweeps, run destructive cleanup, or change unrelated files.

## Documentation Requirements

Keep these files current:

```text
goal-3/status.md
goal-3/jobs.csv
goal-3/findings-summary.md
```

For each phase, create or update:

```text
goal-3/[PHASE-INDEX]-[ONE-WORD-DESCRIPTOR].md
```

Each phase file should include:

- short overview;
- exact implementation steps;
- verification steps;
- completion requirements;
- findings after execution.

For each Slurm job, record:

- job ID;
- script path;
- candidate config;
- seed;
- partition and constraints;
- requested resources;
- host;
- Slurm state and exit code;
- elapsed time;
- artifact path;
- parsed metrics;
- next action.

## Stop Conditions

Stop and ask before continuing if:

- H100/H200 resources are needed and approval has not been given;
- artifact size exceeds 16 MB;
- qMLP breaks the full record stack in a way that needs unplanned H100 debugging;
- FA3 or fused kernels cannot run on OSU H100;
- compliance is ambiguous;
- the exact Slurm request differs from the reviewed request;
- the next action is a broad sweep rather than a bounded run.

## Success Definition

Minimum useful success:

- a qMLP full-stack H100 candidate runs to completion;
- artifact size is under 16 MB;
- BPB, timing, and package metrics are parsed;
- the result is interpretable against the known record stack.

Strong success:

- qMLP `sp16384` beats the local reported `1.06108` H100 record candidate or is
  close enough within seed noise to justify another approved seed.

Full success:

- qMLP produces a compliant, under-16MB, full-stack 8xH100 result that beats the
  known local record by enough margin to treat it as a record-breaking submission
  candidate.
