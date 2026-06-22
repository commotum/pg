Please continue the quaternion Parameter Golf work using the loop defined in `goal/0-loop.md` and the revised roadmap in `goal/0-plan.md`.

The plan has already been redirected after the simple-stack vocabulary ladder. Do not restart from the original simple-baseline framing, and do not treat `qMLP sp4096 > dense sp1024` or `qMLP sp8192 > dense sp1024` as sufficient evidence of a serious winner. Those results are useful leads. The current decision question is:

```text
Does qMLP enable a better best-under-16MB configuration than the known dense/record-stack path?
```

Use `goal/0-loop.md` as the operating procedure and `goal/0-plan.md` as the authoritative current roadmap, but do not treat either file as infallible. Follow current evidence, live cluster state, code behavior, benchmark results, and engineering judgment over stale assumptions. If facts on the ground contradict the plan, update the documents and choose the path that best answers the current decision question.

For each phase in `goal/0-plan.md`:

1. First create a detailed phase plan in `goal/` following the format and naming rules from `goal/0-loop.md`: `[PHASE-INDEX]-[ONE-WORD-DESCRIPTOR].md`.
2. Then implement that phase.
3. Then update `goal/0-plan.md`, `goal/0-loop.md`, the current phase file, and any earlier phase files whose assumptions changed, so the docs match the facts on the ground.
4. Then decide whether to continue, revise, repeat narrowly, block, abandon, or move to the next phase.

Current facts:

- Phase 10 replicated simple-stack qMLP `sp8192`; it is the best simple-stack qMLP candidate so far.
- Phase 11 tested simple-stack qMLP `sp16384` across three A40 seeds; it fit under the package cap but lost to replicated `sp8192`.
- Phase 12 was skipped.
- Phase 13 is now active and should produce the A40 dense record-stack `sp8192` control.
- The active Phase 13 record target is `2026-04-23_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE`, using A40-safe SDPA fallback where FA3 is unavailable.
- The original CaseOps SP8192 data-prep job `20483645` used the old prep path and spent a long time before producing shards. Do not treat it as the primary path without rechecking its state.
- `prepare_caseops_data.py` has been patched to compute validation byte sidecars in a linear pass and to support bounded smoke/full exports with `--max-docs`, `--max-train-shards`, and `--dataset-name`.
- CaseOps smoke-data job `20484885` completed successfully under `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-smoke`, producing one train shard, one validation shard, and one validation-byte sidecar. Use this for the next dense record smoke while preparing the full CaseOps SP8192 data path.
- Phase 14 measures same-vocab record-stack qMLP tax at `sp8192`.
- Phase 15 tests fixed record-stack qMLP `sp16384` as the budget-reinvestment candidate.
- The package-frontier search beyond `sp16384` is deferred. A `11776` canary, job `20484777`, fit easily at `9,376,232` total submission bytes, so `sp16384` is safe enough to benchmark, but do not push to `24576` or `32768` until the fixed A40 comparison says larger vocab is worth pursuing.

Current strategic trajectory:

1. Resume Phase 13 from `goal/13-record.md`.
2. Run the dense record-stack `sp8192` smoke against the completed CaseOps smoke data to validate the A40 record path quickly.
3. Submit or continue a replacement full CaseOps SP8192 export using the patched prep path; only cancel old job `20483645` after verifying the replacement is ahead, complete, or otherwise makes the old job wasteful.
4. Treat missing `flash_attn_interface` on A40 as an SDPA-fallback screening condition, not as an automatic blocker, because the local 04-23 record script has a fallback.
5. After full CaseOps SP8192 data exists, run dense record-stack `sp8192` A40 baseline seeds.
6. Run Phase 14 same-vocab qMLP record-stack `sp8192` smoke and A40 seeds.
7. Run Phase 15 qMLP record-stack `sp16384` package/path smoke and A40 seeds.
8. Compare A40 head-to-head: dense record `sp8192`, qMLP record `sp8192`, and qMLP record `sp16384`.
9. Reopen package-frontier probing only if qMLP record `sp16384` is promising and still has meaningful package headroom.
10. Only if A40 record-stack qMLP results are promising, run H100/FA3 confirmation after explicit review and approval.

Use dense budget controls as a secondary track, not the main path. The dense control question is:

```text
At the simple-stack level, does qMLP beat the best dense configuration that fits under the same artifact cap?
```

Use cheap smokes to estimate dense and qMLP package-size frontiers. Do not waste full A40 benchmark cycles incrementing vocab size one point at a time. Benchmark dense near-budget vocab only if it materially clarifies whether the simple qMLP result was just a larger-vocab effect.

Parallelize independent Slurm jobs when doing so is safe and useful. In particular, independent seed runs should be submitted as a bounded parallel batch when scheduler/account limits allow it, rather than run serially by default. Respect dependency order for export -> smoke/package-size gate -> benchmark, and still ask for approval before large, long, multi-GPU beyond the current approval, destructive, or H100/H200 work.

Keep the work focused on best-under-budget performance. Prefer fast, cheap, well-recorded experiments before scarce or expensive runs. Use OSU/HPC resources safely according to the plan: no training, tokenizer export, GPU diagnostics, or material compute on submit nodes; use Slurm for compute work; record job IDs, commands, logs, artifacts, hardware, seeds, BPB, step counts, memory, and package sizes; ask for approval before large, long, multi-GPU, destructive, or H100/H200 work.

Watch carefully for impossible goals, unverifiable completion requirements, over-strict verification, hardware assumptions that no longer hold, and token-wasting loops. When a requirement is impossible or no longer useful, revise it into the smallest factual test that moves the project forward.

Proceed phase by phase until the revised plan is complete, blocked by a real external condition, or the evidence is strong enough to make a defensible keep/modify/abandon decision on qMLP for best-under-16MB Parameter Golf.
