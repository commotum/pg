# Goal Loop for Quaternion Parameter Golf Experiments

Date drafted: 2026-06-21

## Purpose

This document defines the core work loop for executing `goal/0-plan.md`.

The loop exists to keep the work pointed at one question:

> Does a quaternion MLP help produce lower Parameter Golf validation bits-per-byte after we reinvest the saved parameters?

The loop should produce facts, runnable code, benchmark records, and updated plans. It should not produce endless planning, unverifiable success criteria, or token-expensive work that does not clarify whether the quaternion MLP is useful.

## Core Loop

For each phase in `goal/0-plan.md`, run this sequence.

1. Write the detailed phase plan.
2. Implement that phase.
3. Update all goal documents with new facts from the phase.
4. Move to the next phase.

Repeat until the main question is answered well enough to decide whether to keep, modify, or abandon the quaternion MLP path.

## Phase Plan Files

Before implementing a phase, create a phase-specific Markdown file in `goal/`.

Name each phase file:

```text
[PHASE-INDEX]-[ONE-WORD-DESCRIPTOR].md
```

Examples:

```text
0-setup.md
1-smoke.md
2-baseline.md
3-a40.md
4-qmlp.md
```

The descriptor should be lowercase, one word, and stable enough that it still makes sense after the phase is complete.

Each phase file must contain:

- A short overview of the phase.
- The reason this phase moves us closer to answering the quaternion MLP question.
- Current assumptions and dependencies.
- Detailed implementation steps.
- Commands or scripts to run, where practical.
- Expected artifacts.
- Completion requirements.
- Failure and fallback rules.

The phase file should be specific enough to execute from, but not so rigid that it forces pointless work when the facts change.

## Implementation Pass

After the phase plan exists, implement it.

During implementation:

- Prefer small, reviewable changes.
- Keep remote HPC work on the correct machine type.
- Do not run training, GPU checks, dataset preparation, or material tests on submit nodes.
- Use Slurm allocations for compute work.
- Record commands, job IDs, logs, metrics, and hardware context.
- Avoid broad refactors unless they are required for the phase.
- Treat benchmark results as scientific evidence, not as proof that the current implementation is final.

The implementation pass should end with enough evidence to decide whether the phase completion requirements were met, partially met, or need revision.

## Documentation Update Pass

After implementation, update every goal document that has become stale.

At minimum, check:

- `goal/0-plan.md`
- `goal/0-loop.md`
- The current phase file
- Any earlier phase files whose assumptions were changed by new evidence

Documentation updates should reflect facts on the ground:

- Actual cluster availability, not stale scheduler snapshots.
- Actual dependency and environment state.
- Actual benchmark numbers.
- Actual failed commands and why they failed, if relevant.
- Actual design choices made in code.
- Any changed resource strategy.

Do not rewrite history to make earlier assumptions look correct. Preserve useful failed paths when they teach us something about the experiment.

## Completion Requirements

Every phase must end with completion requirements that are concrete, finite, and useful.

Good completion requirements look like:

- A command completed successfully and produced a named artifact.
- A Slurm job ran on a specified GPU class and produced a log.
- A baseline run produced `val_bpb`, step time, parameter count, and hardware details.
- A qMLP run reached the same reporting format as the baseline.
- A failed run has a captured error and a clear next action.

Bad completion requirements look like:

- "Fully optimize performance."
- "Prove qMLP is best."
- "Make the run production-ready."
- "Match leaderboard performance before moving on."
- "Verify everything."

Completion requirements should be strong enough to prevent sloppy work, but not so strict that they trap the process in an endless loop.

## Impossible Goal Checks

Before starting a phase, and again before declaring it incomplete, check whether any requirement is impossible, unverifiable, or no longer relevant.

Watch for:

- Requirements that depend on unavailable GPUs.
- Requirements that need credentials or access the agent does not have.
- Requirements that require a long or expensive run without human approval.
- Requirements that cannot be measured from available logs.
- Requirements that depend on exact parity between different GPU classes.
- Requirements that require leaderboard-level performance before the basic implementation is known to work.
- Requirements that expand the scope beyond answering whether qMLP helps bits-per-byte.

If a requirement fails this check, revise it into a smaller factual test. Record the revision in the phase file.

## Verification Discipline

Verification should match the risk of the phase.

For setup phases, useful verification may be:

- Repository exists in the expected location.
- Submodules are present.
- Environment creation works.
- A tiny command imports the needed packages.

For implementation phases, useful verification may be:

- Unit or smoke tests pass.
- Parameter counts match expectation.
- Forward and backward passes work.
- A short Slurm job completes.

For benchmark phases, useful verification may be:

- Baseline and qMLP use comparable settings.
- Logs include `val_bpb`, step count, wallclock, GPU type, commit SHA, and configuration.
- Results are saved in a stable location.

Avoid verification that is stricter than the current decision needs. The point is to learn efficiently, not to certify the entire system after every small step.

## Decision Rules

At the end of each phase, make an explicit decision:

- Continue to the next planned phase.
- Repeat the phase with a narrower fix.
- Revise later phases because the facts changed.
- Stop the qMLP path because the evidence is poor.
- Escalate to a better GPU only if the candidate has earned it.

The decision should be based on the current evidence and the main objective: lower `val_bpb` under Parameter Golf constraints.

Parameter savings alone are not success. A qMLP variant matters only if the saved budget can be reinvested into vocabulary, width, depth, or another change that improves bits-per-byte.

## Phase Output Template

Each completed phase should leave behind a short result block in its phase file.

Use this shape:

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

Use `partial` when enough was learned to move forward but some noncritical requirement was not met.

Use `blocked` only when progress genuinely depends on human input, access, hardware availability, or another external condition.

Use `abandoned` when the phase goal is no longer worth doing because a better path or a contradiction was discovered.

## Token and Time Guardrails

When work starts to loop, stop and reduce the scope.

Signs of a bad loop:

- Rechecking the same static file without new reason.
- Rewriting plans without new facts.
- Running more benchmarks without a decision they can change.
- Chasing exact reproducibility before basic signal exists.
- Expanding from qMLP evaluation into unrelated architecture work.
- Treating every failed command as a reason for broad environment redesign.

The preferred response to a bad loop is:

1. State the smallest unresolved question.
2. Identify the cheapest command or code change that answers it.
3. Run that test or record why it cannot be run.
4. Update the phase file.
5. Decide whether to continue.

## Main Success Standard

The whole loop succeeds when we can make a defensible call on the quaternion MLP idea.

The strongest positive outcome is:

- qMLP trains correctly.
- qMLP preserves or improves speed enough to be practical.
- qMLP frees enough parameters to increase vocabulary, width, depth, or another useful component.
- The reinvested qMLP configuration improves `val_bpb` against a comparable baseline.
- The result survives at least one higher-quality confirmation run.

A useful negative outcome is also acceptable:

- qMLP trains correctly but does not improve `val_bpb`.
- qMLP is too slow for the competition wallclock.
- qMLP destabilizes training.
- Reinvesting saved parameters elsewhere consistently performs worse than the dense baseline.

Either outcome is progress if it is backed by clear implementation facts and benchmark evidence.
