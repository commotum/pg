# Goal Loop for Lean A40 Parameter Golf Experiments

Date drafted: 2026-06-22

## Purpose

This document defines the work loop for `goal-2/0-plan.md`.

The previous goal found useful qMLP and record-stack facts, but it also spent too much time on full phased TTT on A40. That was a poor fit for the hardware. This loop is redirected around lean, single-GPU A40 exploration.

The current question is:

```text
Can we build an A40-friendly Parameter Golf screening stack from common successful record-track ideas, then test whether qMLP improves best-under-16MB BPB across CaseOps vocab sizes?
```

This is not a final record-compliance loop. It is a fast experimental loop for deciding which ideas deserve scarce H100/FA3 confirmation later.

## Core Loop

For each phase in `goal-2/0-plan.md`:

1. Create a detailed phase plan in `goal-2/`.
2. Implement the phase.
3. Update `goal-2/0-plan.md`, this loop file if needed, the current phase file, and any earlier phase files whose assumptions changed.
4. Decide whether to continue, revise, repeat narrowly, stop, or escalate to a better GPU.

Do not continue old `goal-1` phases unless the user explicitly asks.

## Non-Sequential Work

Treat phase order as a dependency graph, not a blocking queue. Do not wait idly
for one phase or one vocab/model cell when independent useful work is available.

Allowed non-sequential work:

- verify completed CaseOps exports while other exports are still running;
- run package/path smokes for ready vocab/model cells;
- run A40 benchmarks for cells whose data and smoke gates have passed;
- submit dependent Slurm jobs with `afterok` when the dependency is clear;
- patch harness/parser/docs while compute jobs run;
- parse completed runs and update matrix summaries before all cells finish.

Do not bypass cell-level gates. Each `(model_variant, vocab_size, seed)` cell
still needs:

```text
verified CaseOps data -> package/path smoke -> benchmark -> parsed metrics
```

When working out of numeric phase order, record why in the active phase file and
update the plan if the dependency graph changes.

## Phase Files

Each phase plan must be named:

```text
[PHASE-INDEX]-[ONE-WORD-DESCRIPTOR].md
```

Examples:

```text
0-inventory.md
1-data.md
2-harness.md
3-baseline.md
4-qmlp.md
5-ladder.md
```

Each phase file should include:

- short overview;
- why the phase answers the A40/qMLP question;
- current assumptions and dependencies;
- implementation steps;
- expected artifacts;
- completion requirements;
- failure and fallback rules;
- result block after work completes.

## Lean/Fat Guardrail

Before starting a phase, classify the work.

Lean work is appropriate for repeated A40 exploration:

- CaseOps tokenizer/data prep and original-byte BPB checks;
- single-GPU SDPA/eager training and eval;
- no-TTT or tiny TTT legality smokes;
- qMLP correctness and matched BPB comparisons;
- XSA, sparse gates, attention output gates, SmearGate, BigramHash, partial RoPE, QK gain;
- package-size, quantization, and compression checks that finish quickly;
- bounded seed batches.

Fat work is not part of the A40 loop unless explicitly approved:

- full phased TTT over the validation set;
- long-context TTT;
- H100/Hopper-only FA3 performance paths;
- 8xH100 multi-GPU reproduction;
- full record-stack schedule tuning that depends on thousands of 8xH100 steps;
- lrzip/per-group compression if serialization dominates iteration time;
- broad hyperparameter searches over full 600-second runs.

Fat work may receive a correctness smoke, but not a long A40 benchmark, unless the user explicitly redirects.

## Primary Metrics

Use these as the default A40 screening metrics:

- post-quant no-TTT `val_bpb`;
- pre-quant `val_bpb`;
- package size under the 16,000,000-byte cap;
- steps completed in the chosen A40 wallclock;
- milliseconds per step;
- peak GPU memory;
- seed variance for serious candidates.

Full TTT BPB is not a default A40 metric. It is only a legality or implementation smoke unless moved to H100/FA3 with a reviewed command.

## Implementation Rules

- Do not run training, tokenizer export, GPU checks, or material compute on submit nodes.
- Use Slurm for compute work.
- Use CPU Slurm allocations for tokenizer/data export.
- Use GPU Slurm allocations for training/eval/package smokes.
- Parallelize independent jobs when safe, especially seed batches, vocab smokes, and package probes.
- Preserve dependency order: data export -> package/path smoke -> benchmark.
- Record job IDs, commands, paths, seeds, git SHAs, BPB, package size, step count, memory, host, and failure modes.
- Keep generated logs in run directories, not in the source tree.
- Do not request H100/H200, multi-GPU, more than 16 CPUs, more than 128 GB RAM, or more than 8 hours without explicit approval.

## Completion Requirements

Every phase must have finite, testable completion requirements.

Good requirements:

- a named Slurm job reaches terminal state;
- a tokenizer/data export produces model, vocab, train shards, validation shards, and byte sidecars;
- a package smoke reports total submission bytes;
- a benchmark reports pre-quant, post-quant, no-TTT BPB, steps, step time, and memory;
- a failure is captured with enough detail to revise the next action.

Bad requirements:

- optimize fully;
- match the leaderboard;
- prove qMLP is best;
- verify everything;
- keep tuning until it works.

## Decision Rules

At the end of each phase, make one explicit decision:

- continue to the next planned phase;
- repeat with a narrower fix;
- revise later phases because a fact changed;
- abandon a path because it is fat or unpromising;
- escalate to H100/FA3 only if lean A40 evidence justifies it.

qMLP success means lower BPB under a fair matched A40 screening setup and a plausible path under the 16 MB artifact cap. Parameter savings alone are not success.

## Result Template

Use this result block at the end of each phase file:

```markdown
## Result

Status: complete | partial | blocked | abandoned

Evidence:

- ...

Artifacts:

- ...

New facts:

- ...

Decision:

- ...
```
