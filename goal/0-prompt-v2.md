# Prompt V2: Redirect Quaternion Parameter Golf Goal

You are continuing the work governed by `goal/0-plan.md` and `goal/0-loop.md`.

Your first task is to edit `goal/0-plan.md` so the plan reflects the revised strategy below. Do not start new Slurm jobs or benchmarks until `goal/0-plan.md` has been updated.

Important context:

- Phases 0-9 are already useful and should not be discarded.
- The key positive result through Phase 8 was that matrix qMLP plus `sp4096` beat the dense `sp1024` simple baseline across three A40 seeds.
- Phase 9 then showed matrix qMLP plus `sp8192` was stronger still on seed 42, stayed under the 16 MB artifact cap, and is now the best simple-stack qMLP candidate so far.
- That result proves qMLP can support a useful budget reinvestment, but it does not yet prove qMLP is competitive against a dense model that also spends the 16 MB artifact budget intelligently.
- The plan should finish the immediate simple-stack qMLP vocab checks already in flight, then pivot toward record-stack relevance and budget-matched record-stack controls.

Update `goal/0-plan.md` accordingly before running more benchmarks. This prompt is not asking for a separate analysis memo; it is asking you to revise the plan document itself.

## Required Plan Change

Revise the next phases around this decision question:

> Does qMLP enable a better best-under-16MB configuration than the known dense/record-stack path?

Do not keep treating `qMLP sp4096 > dense sp1024` as sufficient evidence for a serious winner. It is a good lead, not the final control.

## New Strategic Track

The new trajectory should be:

1. Finish `sp8192` qMLP seed replication on the simple stack.
   - Run at least two additional A40 seeds using the exact Phase 9 `sp8192` benchmark shape.
   - Compare against the Phase 8 `sp4096` mean and dense `sp1024`.
   - Track steps, BPB, artifact size, memory, host, and job IDs.
   - If `sp8192` is seed-robust, keep it as the current simple-stack qMLP candidate.

2. Probe `sp16384` qMLP on the simple stack.
   - Export bounded `sp16384` data/tokenizer if needed.
   - Run a cheap smoke/package-size gate before any 10-minute benchmark.
   - If the package is safely under 16 MB, run one seed-42 A40 10-minute benchmark.
   - If it improves over `sp8192` or is close enough to be plausible, seed-replicate it with at least two more seeds.
   - If it is over budget or clearly worse, stop the simple-stack vocab ladder and move to record-stack work.

3. Reproduce the strongest manageable current record-setting stack as-is on A40.
   - Prefer the local record stack with CaseOps/special vocab and the known optimizations if it can be made to run without spending days on H100-only dependencies.
   - Capture BPB, steps, ms/step, artifact size, model/config, tokenizer/data path, exact hardware, and exact command.
   - Treat A40 as a screening/debugging environment, not final proof.
   - Run a few seeds, not just one, so the comparison has a stable read.

4. Add qMLP to that same record-stack configuration without changing vocab or unrelated settings.
   - This is expected to be worse.
   - The purpose is to measure the qMLP tax inside the strong stack:

   ```text
   net_gain = benefit_from_reinvested_budget - qMLP_expressiveness_or_training_tax
   ```
   
   - Run enough seeds to estimate the tax rather than overfitting to a single lucky or unlucky run.

5. Use qMLP's saved package/model budget inside the record stack.
   - Push vocab/package size close to the 16 MB cap with cheap smoke/package-size probes rather than full benchmark cycles at every vocab size.
   - Test near-frontier candidates such as power-of-two and near-cap vocab sizes, but avoid invalidating runs by cutting the budget too close.
   - The goal is the best record-stack qMLP configuration under 16 MB, not maximum vocab for its own sake.
   - Benchmark only serious candidates after package-size smoke passes.

6. Compare A40 head-to-head:
   - original record stack as-is;
   - same-vocab record stack with qMLP, to measure qMLP tax;
   - vocab-max or budget-reinvested qMLP record stack.
   - Use multiple seeds where feasible for the original record stack and final qMLP contender.

7. Only if the A40 record-stack qMLP result is promising, confirm on the intended H100/FA3 path.
   - First run a small H100 compatibility/speed check.
   - Then run the relevant 8xH100/FA3 test only after the A40 evidence justifies it and the exact command/config is reviewed.
   - Do not spend scarce H100/H200 resources merely to continue simple-stack exploration.

## Dense Budget Controls

Add this as a secondary control track, not the main path:

- Use cheap smokes to find approximately how much vocab dense and qMLP variants can fit under 16 MB.
- Do not waste full A40 benchmark cycles incrementing one vocab size at a time.
- Benchmark dense near-budget vocab only if it materially clarifies whether the simple qMLP result was just a larger-vocab effect.

The dense control question is:

> At the simple-stack level, does qMLP beat the best dense configuration that fits under the same artifact cap?

## Documentation Requirements

When updating `goal/0-plan.md`:

- Preserve the completed Phase 0-8 result summaries.
- Preserve the completed Phase 9 result if it is already present.
- Mark the old Phase 10+ trajectory as revised, not erased.
- Add explicit phases for:
  - `sp8192` seed replication;
  - `sp16384` initial qMLP probe;
  - `sp16384` seed replication if it earns replication;
  - record-stack inventory/reproduction;
  - same-vocab qMLP tax measurement;
  - record-stack qMLP budget-frontier vocab/package probe;
  - A40 head-to-head comparison;
  - 8xH100/FA3 confirmation only after A40 record-stack success.
- Update decision gates so they test best-under-budget performance, not qMLP versus an underfilled dense baseline.
- Keep the existing operating rules: no training on submit nodes, use Slurm, record job IDs/logs/artifacts, and avoid scarce H100/H200 until earned.
