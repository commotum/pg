# Goal 3 Loop

This loop governs the 8xH100 qMLP record attempt described in
`goal-3/0-plan.md`.

## Core Rule

Do not use scarce H100 time for open-ended exploration. H100 work should execute
prewritten scripts that have already passed non-scarce checks.

Each phase follows this loop:

1. Write or update the phase plan as:

```text
goal-3/[PHASE-INDEX]-[ONE-WORD-DESCRIPTOR].md
```

2. Implement only the work needed for that phase.
3. Run the cheapest meaningful verification for that phase.
4. Update all Goal 3 docs that are now stale.
5. Update `goal-3/status.md` and `goal-3/jobs.csv`.
6. Move to the next phase only when the completion requirements are satisfied
   or the phase is explicitly revised.

## Non-Sequential Work

Do useful ready work in parallel when dependency order allows it.

Examples:

- While data manifests are being verified, inspect record-stack code and draft
  Slurm scripts.
- While a CPU Slurm export runs, implement parsers or qMLP port checks.
- While a non-H100 smoke is queued, update docs and static-check the H100
  runner.

Do not wait idly for one phase artifact if another safe prerequisite can move.
Do not bypass gates that protect H100 time.

## Facts Beat Stale Plans

Plans are working documents. If live scheduler output, logs, package sizes, code
inspection, or compliance facts contradict the plan:

1. trust the new evidence;
2. update the relevant docs;
3. explain the change in `goal-3/status.md`;
4. continue with the revised path.

Do not keep pursuing a stale phase just because it was written earlier.

## Impossible Goal Guard

Watch for requirements that are impossible, unverifiable, or likely to cause an
endless loop.

Bad requirements:

- "prove qMLP always wins";
- "match the record exactly" without the same hardware/software;
- "optimize until best" without a bounded search space;
- "verify compliance" without naming the specific compliance checks.

Replace them with bounded requirements:

- one run reaches terminal Slurm state;
- package size is parsed and under cap;
- exact command and artifact path are recorded;
- a specific smoke passes or fails with logs;
- a result beats or does not beat a named baseline.

## Submit-Node Boundary

Allowed on submit nodes:

- edit files;
- inspect small files;
- run Git;
- inspect Slurm state;
- submit, inspect, or cancel specifically identified jobs;
- run static checks such as `bash -n`.

Use Slurm compute allocations for:

- training;
- eval;
- package smokes;
- FA3/Triton compilation;
- tokenizer/data export;
- GPU checks;
- sustained CPU or I/O work.

If unsure, use a small Slurm allocation.

## H100 Approval Gate

No H100 or H200 job may be submitted until the user has reviewed and approved:

- exact script path;
- exact Slurm resources;
- requested walltime;
- candidate run order;
- stop conditions;
- artifact and log destination;
- known risks.

The approval must happen after the final script and live scheduler check are
available. Old approval for a different script or different resource request
does not carry over automatically.

## H100 Execution Loop

Inside the H100 allocation, the default runner should be deterministic:

1. record hardware and environment;
2. validate or build required runtime pieces, including FA3 when allowed;
3. run H100/FA3/NCCL/tokenizer smoke checks;
4. run the full dense/base `sp8192` baseline parity candidate;
5. stop if baseline parity fails the reviewed BPB, step-count, exit-code, or
   artifact-size gates;
6. run the predeclared qMLP `sp16384` seeds;
7. validate package size and parse metrics for every completed run;
8. copy logs, full models, quantized artifacts, hashes, summaries, and final
   status to shared storage.

If a step fails, the runner should stop unless a specific fallback was
predeclared.

## Codex-On-Node Policy

Codex can be used on the HPC side only as a bounded fallback, not as the primary
way to spend H100 time.

Allowed:

- `codex exec` in noninteractive mode;
- strict prompt from `goal-3/0-prompt.md`;
- `timeout`;
- workspace-write access to the project/run directory;
- diagnosis and small patching inside the current allocation;
- one bounded smoke after a patch, if the prompt permits it.

Not allowed:

- open-ended exploration while 8 GPUs sit allocated;
- submitting new H100 jobs;
- broad sweeps;
- destructive cleanup;
- changing unrelated files;
- bypassing the Goal 3 plan and status docs.

The normal path should be a deterministic Slurm runner. The repair agent exists
only for H100-specific failures that could not be found earlier.

## Documentation Updates

After every meaningful action, update the relevant docs.

Always keep these current:

```text
goal-3/status.md
goal-3/jobs.csv
goal-3/findings-summary.md
```

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

## Result Interpretation

Separate these effects:

- qMLP architecture effect at the same vocab;
- vocab reinvestment effect;
- throughput/step-count effect;
- quantization/package effect;
- TTT/eval effect;
- hardware/software mismatch effect.

Do not claim a record-level win from an A40 result. Do not claim qMLP is useful
in the full record stack until the full stack runs.

## Stop Or Continue

Continue when:

- the next phase has clear completion requirements;
- the required work is bounded;
- the result will materially answer the qMLP record question.

Stop and ask when:

- H100 resources are required and not yet approved;
- package compliance is ambiguous;
- the next action is a broad sweep;
- a required dependency is unavailable;
- a result invalidates the planned path;
- further work would spend scarce compute without improving the decision.
