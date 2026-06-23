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
- CaseOps smoke-data job `20484885` completed successfully under `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-smoke`, producing one train shard, one validation shard, and one validation-byte sidecar.
- Dense record pre-quant smoke job `20484894` completed successfully on A40 smoke data with SDPA fallback, `DOCUMENT_PACKING=0`, `TORCH_COMPILE=0`, `FUSED_MLP_ENABLED=0`, `TTT_ENABLED=0`, and `PREQUANT_ONLY=1`.
- Dense record package-path smoke job `20484900` completed on A40 smoke data with exit code `0:0`, reporting quantized+brotli model size `15881408` bytes and total submission size `15913223` bytes.
- Patched full CaseOps SP8192 export job `20484895` completed under `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched` with `MAX_TRAIN_SHARDS=80`, `VAL_DOCS=10000`, and 2 CPUs. It produced 80 train shards, one validation shard, and one validation-byte sidecar in `01:04:36`. Old unpatched job `20483645` was canceled after the patched replacement produced shards while the old job still had none.
- Dense record A40 baseline jobs at `TRAIN_BATCH_TOKENS=786432` OOMed, and a `524288` fit smoke also OOMed. Fit smoke job `20485067` completed at `TRAIN_BATCH_TOKENS=262144`.
- Dense record A40 baseline jobs ran at `TRAIN_BATCH_TOKENS=262144`: seed `42` job `20485084`, seed `0` job `20485085`, and seed `1` job `20485086`. All three reached 65 training steps at the 600-second train cap, produced under-cap total submission sizes around `15.947-15.949 MB`, then failed after diagnostic quantized eval with `UnboundLocalError` from deleting `eval_model` before the TTT branch.
- Phase 14 qMLP jobs `20485087`, `20485088`, `20485089`, and `20485090` were canceled while still pending because their captured script default pointed at the older `caseops-sp8192` data root and the submitted environment override was not visible.
- Phase 13/14 script defaults now point at `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched`.
- Replacement Phase 14 qMLP jobs `20485171`, `20485172`, `20485173`, and `20485174` were canceled after dense seed `20485084` failed and their `afterok` dependency could never be satisfied.
- The 04-23 record `train_gpt.py` cleanup path was patched and passed local and remote `py_compile`.
- Repaired dense seed `42` job `20485200` proved the cleanup fix by passing diagnostic quantized eval and package output, but failed during TTT compile with CUDA OOM at the record default `TTT_BATCH_SIZE=64`.
- Dependent follow-up jobs `20485214`-`20485219` were canceled after `20485200` failed.
- Dense/qMLP A40 runners now expose/default `TTT_BATCH_SIZE=32`.
- TTT-only repair job `20485290` completed successfully against the existing `20485200` quantized artifact with `TTT_EVAL_ONLY=1` and `TTT_BATCH_SIZE=32`, producing final `quantized_ttt_phased val_bpb=2.48394114`.
- Gated follow-up jobs were queued without bypassing the Phase 13 gate: dense seed `0` `20485344` and dense seed `1` `20485345` depended on `afterok:20485290`; qMLP smoke `20485346` depended on `afterok:20485290`; qMLP seed jobs `20485348`, `20485349`, and `20485350` depended on `afterok:20485346`.
- `20485290` reached `ttpr: phase:1/3 t:1975.1s`, so full phased TTT may exceed the original `01:30:00` A40 limit. Slurm denied extending the running repair, but queued full seed jobs `20485344`, `20485345`, `20485348`, `20485349`, and `20485350` were extended to `02:30:00`; dependency checks still showed the intended `afterok` gates.
- `20485290` completed successfully in `01:01:45`, producing `quantized_ttt_phased val_loss:5.33687296 val_bpb:2.48394114 eval_time:3407345ms` and `total_eval_time:3407.3s`. Dense seed `42` should be treated as a combined repaired result from `20485200` plus `20485290`.
- The successful gate released dense seed jobs `20485344` and `20485345` plus Phase 14 qMLP smoke `20485346`; after the qMLP smoke passed, qMLP seed jobs `20485348`, `20485349`, and `20485350` were released and completed.
- Phase 14 qMLP smoke `20485346` completed successfully in `00:06:57`, reporting `model_params:18644154`, step-2 `val_bpb:4.3033`, post-EMA pre-quant `val_bpb:4.17302032`, and max RSS about `3.97 GB`.
- Phase 13 dense record `sp8192` A40 control completed across three final-TTT seeds:
  - seed `42`: combined jobs `20485200` + `20485290`, total submission size `15,949,143` bytes, diagnostic quantized `val_bpb=3.63838697`, final `val_bpb=2.48394114`;
  - seed `0`: job `20485344`, total submission size `15,947,462` bytes, diagnostic quantized `val_bpb=3.69540335`, final `val_bpb=2.48788026`;
  - seed `1`: job `20485345`, total submission size `15,948,080` bytes, diagnostic quantized `val_bpb=3.65207914`, final `val_bpb=2.48370635`;
  - mean final TTT BPB `2.48517592`, mean total submission size about `15,948,228` bytes.
- Phase 14 qMLP record `sp8192` A40 completed across three final-TTT seeds:
  - seed `42`: job `20485348`, total submission size `8,465,062` bytes, diagnostic quantized `val_bpb=2.98003322`, final `val_bpb=2.28242806`;
  - seed `0`: job `20485349`, total submission size `8,465,567` bytes, diagnostic quantized `val_bpb=2.98579482`, final `val_bpb=2.28778189`;
  - seed `1`: job `20485350`, total submission size `8,464,761` bytes, diagnostic quantized `val_bpb=2.98862174`, final `val_bpb=2.28335877`;
  - mean final TTT BPB `2.28452291`, mean total submission size about `8,465,130` bytes.
- Same-vocab qMLP tax was negative on the A40 screening setup: `2.28452291 - 2.48517592 = -0.20065301 BPB`.
- Phase 15 CaseOps `sp16384` prep job `20484985` completed in `00:50:28` with bounded `MAX_TRAIN_SHARDS=80` and 16 CPUs. File verification found 80 train shards, one validation shard, and one validation-byte sidecar.
- Phase 14 is complete, and qMLP `sp8192` is the current best A40 record-stack candidate.
- Phase 15 now tests fixed record-stack qMLP `sp16384` as the budget-reinvestment candidate, asking whether larger vocab improves on qMLP `sp8192`.
- The package-frontier search beyond `sp16384` is deferred. A `11776` canary, job `20484777`, fit easily at `9,376,232` total submission bytes, so `sp16384` is safe enough to benchmark, but do not push to `24576` or `32768` until the fixed A40 comparison says larger vocab is worth pursuing.

Current strategic trajectory:

1. Start Phase 15 from `goal/15-sp16384.md`.
2. Use completed patched full CaseOps `sp16384` export `20484985` as the Phase 15 data path.
3. Treat Phase 13 dense record `sp8192` and Phase 14 qMLP record `sp8192` A40 controls as complete.
4. Treat missing `flash_attn_interface` on A40 as an SDPA-fallback screening condition, and keep `DOCUMENT_PACKING=0`, `TORCH_COMPILE=0`, `FUSED_MLP_ENABLED=0`, `TRAIN_BATCH_TOKENS=262144`, and `TTT_BATCH_SIZE=32` consistent across the A40 record runs unless a later paired smoke proves a different compatibility set works.
5. Run Phase 15 qMLP record-stack `sp16384` package/path smoke and A40 seeds in parallel where scheduler/account limits allow.
6. Compare A40 head-to-head: dense record `sp8192`, qMLP record `sp8192`, and qMLP record `sp16384`.
7. Reopen package-frontier probing only if qMLP record `sp16384` is promising and still has meaningful package headroom.
8. Only if A40 record-stack qMLP results are promising, run H100/FA3 confirmation after explicit review and approval.

Use dense budget controls as a secondary track, not the main path. The dense control question is:

```text
At the simple-stack level, does qMLP beat the best dense configuration that fits under the same artifact cap?
```

Use cheap smokes to estimate dense and qMLP package-size frontiers. Do not waste full A40 benchmark cycles incrementing vocab size one point at a time. Benchmark dense near-budget vocab only if it materially clarifies whether the simple qMLP result was just a larger-vocab effect.

Parallelize independent Slurm jobs when doing so is safe and useful. In particular, independent seed runs should be submitted as a bounded parallel batch when scheduler/account limits allow it, rather than run serially by default. Respect dependency order for export -> smoke/package-size gate -> benchmark, and still ask for approval before large, long, multi-GPU beyond the current approval, destructive, or H100/H200 work.

Keep the work focused on best-under-budget performance. Prefer fast, cheap, well-recorded experiments before scarce or expensive runs. Use OSU/HPC resources safely according to the plan: no training, tokenizer export, GPU diagnostics, or material compute on submit nodes; use Slurm for compute work; record job IDs, commands, logs, artifacts, hardware, seeds, BPB, step counts, memory, and package sizes; ask for approval before large, long, multi-GPU, destructive, or H100/H200 work.

Watch carefully for impossible goals, unverifiable completion requirements, over-strict verification, hardware assumptions that no longer hold, and token-wasting loops. When a requirement is impossible or no longer useful, revise it into the smallest factual test that moves the project forward.

Proceed phase by phase until the revised plan is complete, blocked by a real external condition, or the evidence is strong enough to make a defensible keep/modify/abandon decision on qMLP for best-under-16MB Parameter Golf.
