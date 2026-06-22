# Prompt V2: Redirect Quaternion Parameter Golf Goal

You are continuing the work governed by `goal/0-plan.md` and `goal/0-loop.md`.

Your first task is to edit `goal/0-plan.md` so the plan reflects the revised strategy below. Do not start new Slurm jobs or benchmarks until `goal/0-plan.md` has been updated.

Important context:

- Phases 0-8 are already useful and should not be discarded.
- The key positive result so far is that matrix qMLP plus `sp4096` beat the dense `sp1024` simple baseline across three A40 seeds.
- That result proves qMLP can support a useful budget reinvestment, but it does not yet prove qMLP is competitive against a dense model that also spends the 16 MB artifact budget intelligently.
- The plan now needs to pivot from simple-baseline exploration toward record-stack relevance and budget-matched controls.

Update `goal/0-plan.md` accordingly before running more benchmarks. This prompt is not asking for a separate analysis memo; it is asking you to revise the plan document itself.

## Required Plan Change

Revise the next phases around this decision question:

> Does qMLP enable a better best-under-16MB configuration than the known dense/record-stack path?

Do not keep treating `qMLP sp4096 > dense sp1024` as sufficient evidence for a serious winner. It is a good lead, not the final control.

## New Strategic Track

The new trajectory should be:

1. Reproduce the strongest manageable current record stack as-is on A40.
   - Prefer the local record stack with CaseOps/special vocab and the known optimizations if it can be made to run without spending days on H100-only dependencies.
   - Capture BPB, steps, ms/step, artifact size, model/config, tokenizer/data path, and exact hardware.
   - Treat A40 as a screening/debugging environment, not final proof.

2. Add qMLP to that same record-stack configuration without changing vocab or unrelated settings.
   - This is expected to be worse.
   - The purpose is to measure the qMLP tax inside the strong stack:

   ```text
   net_gain = benefit_from_reinvested_budget - qMLP_expressiveness_or_training_tax
   ```

   - Do not abandon qMLP merely because same-vocab qMLP loses; abandon only if the tax is so large that plausible budget reinvestment cannot recover it.

3. Use qMLP's saved package/model budget inside the record stack.
   - Push vocab/package size close to the 16 MB cap with cheap smoke/package-size probes rather than full benchmark cycles at every vocab size.
   - Test near-frontier candidates such as power-of-two and near-cap vocab sizes, but avoid invalidating runs by cutting the budget too close.
   - Benchmark only serious candidates after package-size smoke passes.

4. Compare A40 head-to-head:
   - original record stack as-is;
   - same-vocab record stack with qMLP, to measure qMLP tax;
   - vocab-max or budget-reinvested qMLP record stack.

5. Only if the A40 result is promising, confirm on the intended H100/FA3 path.
   - First run a small H100 compatibility/speed check.
   - Only request large H100/H200 resources after the A40 evidence justifies it and the exact command/config is reviewed.

## Dense Budget Controls

Also add a smaller control track for the simple baseline:

- Use cheap smokes to find approximately how much vocab dense and qMLP variants can fit under 16 MB.
- Do not waste full A40 benchmark cycles incrementing one vocab size at a time.
- Benchmark dense near-budget vocab only if it materially clarifies whether the simple qMLP result was just a larger-vocab effect.

The dense control question is:

> At the simple-stack level, does qMLP beat the best dense configuration that fits under the same artifact cap?

## Documentation Requirements

When updating `goal/0-plan.md`:

- Preserve the completed Phase 0-8 result summaries.
- Mark the old Phase 9+ trajectory as revised, not erased.
- Add explicit phases for:
  - record-stack inventory/reproduction;
  - same-vocab qMLP tax measurement;
  - record-stack qMLP budget-frontier vocab/package probe;
  - A40 head-to-head comparison;
  - H100/FA3 confirmation only after A40 success.
- Update decision gates so they test best-under-budget performance, not qMLP versus an underfilled dense baseline.
- Keep the existing operating rules: no training on submit nodes, use Slurm, record job IDs/logs/artifacts, and avoid scarce H100/H200 until earned.

## Tone of the Updated Plan

Be direct and skeptical.

The current result is a real lead, but not yet a serious winner. The plan should now be designed to falsify it quickly against stronger controls.
