Please continue the quaternion Parameter Golf work using the loop defined in `goal/0-loop.md` and the revised roadmap in `goal/0-plan.md`.

The plan has already been redirected after Phase 9. Do not restart from the original simple-baseline framing, and do not treat `qMLP sp4096 > dense sp1024` as sufficient evidence of a serious winner. That result is a useful lead. The current decision question is:

```text
Does qMLP enable a better best-under-16MB configuration than the known dense/record-stack path?
```

Use `goal/0-loop.md` as the operating procedure and `goal/0-plan.md` as the authoritative current roadmap, but do not treat either file as infallible. Follow current evidence, live cluster state, code behavior, benchmark results, and engineering judgment over stale assumptions. If facts on the ground contradict the plan, update the documents and choose the path that best answers the current decision question.

For each phase in `goal/0-plan.md`:

1. First create a detailed phase plan in `goal/` following the format and naming rules from `goal/0-loop.md`: `[PHASE-INDEX]-[ONE-WORD-DESCRIPTOR].md`.
2. Then implement that phase.
3. Then update `goal/0-plan.md`, `goal/0-loop.md`, the current phase file, and any earlier phase files whose assumptions changed, so the docs match the facts on the ground.
4. Then decide whether to continue, revise, repeat narrowly, block, abandon, or move to the next phase.

Current strategic trajectory:

1. Finish `sp8192` qMLP seed replication on the simple stack.
2. Probe `sp16384` qMLP with a cheap smoke/package-size gate before any full benchmark.
3. Seed-replicate `sp16384` only if it earns replication.
4. Stop the simple-stack vocabulary ladder after `sp16384`.
5. Reproduce the strongest manageable current record-setting stack as-is on A40.
6. Add qMLP to that same record-stack configuration without changing vocab or unrelated settings, to measure qMLP tax.
7. Use qMLP's saved package/model budget inside the record stack with smoke/package-size probes near the 16 MB frontier.
8. Compare A40 head-to-head: original record stack, same-vocab qMLP record stack, and budget-reinvested qMLP record stack.
9. Only if A40 record-stack qMLP results are promising, run H100/FA3 confirmation.

Use dense budget controls as a secondary track, not the main path. The dense control question is:

```text
At the simple-stack level, does qMLP beat the best dense configuration that fits under the same artifact cap?
```

Use cheap smokes to estimate dense and qMLP package-size frontiers. Do not waste full A40 benchmark cycles incrementing vocab size one point at a time. Benchmark dense near-budget vocab only if it materially clarifies whether the simple qMLP result was just a larger-vocab effect.

Keep the work focused on best-under-budget performance. Prefer fast, cheap, well-recorded experiments before scarce or expensive runs. Use OSU/HPC resources safely according to the plan: no training, tokenizer export, GPU diagnostics, or material compute on submit nodes; use Slurm for compute work; record job IDs, commands, logs, artifacts, hardware, seeds, BPB, step counts, memory, and package sizes; ask for approval before large, long, multi-GPU, destructive, or H100/H200 work.

Watch carefully for impossible goals, unverifiable completion requirements, over-strict verification, hardware assumptions that no longer hold, and token-wasting loops. When a requirement is impossible or no longer useful, revise it into the smallest factual test that moves the project forward.

Proceed phase by phase until the revised plan is complete, blocked by a real external condition, or the evidence is strong enough to make a defensible keep/modify/abandon decision on qMLP for best-under-16MB Parameter Golf.
