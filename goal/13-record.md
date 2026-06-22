# Phase 13: Record-Stack Inventory And A40 Reproduction

Date drafted: 2026-06-22

Status: active. Phase 11 completed, Phase 12 was skipped, and the plan now moves from simple-stack vocabulary work to record-stack A40 controls.

## Overview

Reproduce the strongest manageable current record-setting stack as-is on A40 before adding qMLP.

This phase exists to establish the relevant dense/record-stack control. The project question is no longer whether simple qMLP beats an underfilled dense `sp1024` baseline. The current question is whether qMLP can help produce a best-under-16MB configuration against the known dense/record-stack path.

## Why This Matters

Simple-stack qMLP plus larger vocabulary is a useful lead, but record-stack performance is much stronger. A qMLP result matters only if it can beat, match, or create a credible path beyond the strongest manageable record-stack control under the same package budget.

The record-stack reproduction should answer:

- Which local record stack can actually run on OSU A40 within bounded setup effort?
- What BPB, throughput, memory, artifact size, and failure modes does that stack show on A40?
- Is the chosen control stable enough to measure same-vocab qMLP tax in Phase 14?

## Current Candidate Inventory

Read-only inventory on 2026-06-22 identified the likely candidates under `parameter-golf/records/track_10min_16mb/`.

Best published local record:

```text
path=parameter-golf/records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611
mean_val_bpb=1.06107587
seeds=42,0,1234
hardware=8xH100 80GB SXM
artifact_bytes_max=15907550
steps_mean=4931.33
step_avg_ms_mean=121.7
requirements=PyTorch 2.9.1+cu128, CUDA 12.8, FlashAttention 3, lrzip
notes=best BPB but includes per-group lrzip compression and tight H100/FA3 assumptions
```

Strong compliance reproduction:

```text
path=parameter-golf/records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141
mean_val_bpb=1.06145
rerun_mean_bpb=1.06141
seeds=42,314,1234
hardware=8xH100 SXM 80GB
artifact_bytes_max=15952690
requirements=PyTorch 2.9+, FlashAttention interface, CaseOps SP8192
notes=no new technique; useful compliance reproduction of PR #1851, but artifact headroom is only about 47 KB
```

Slightly older strong stack:

```text
path=parameter-golf/records/track_10min_16mb/2026-04-23_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE
mean_val_bpb=1.06335
seeds=42,0,1234
hardware=8xH100 80GB SXM
artifact_bytes_max=15940380
requirements=PyTorch 2.9.1+cu128, flash-attn-interface, triton, sentencepiece
notes=slightly worse than 04-27/04-29, but simpler than the 04-27 lrzip/pergroup compression path
```

The inspected record stacks are H100/FA3-oriented and the full reproduction commands use `torchrun --standalone --nproc_per_node=8`. The chosen 04-23 A40 screening path has been locally patched to tolerate missing `flash_attn_interface` by falling back to PyTorch SDPA. Therefore A40 reproduction is expected to use one of:

- an environment where the imported FlashAttention interface works on A40;
- the bounded SDPA fallback currently present in the local 04-23 script; or
- choosing a lower-ranked record-style control that is already A40-compatible.

## Working Assumptions

- Phase 13 begins now because Phase 11 ended the simple-stack vocabulary ladder and Phase 12 was skipped.
- A40 is a screening/debugging environment, not final proof of leaderboard performance.
- No H100/H200 job should be used merely to continue simple-stack exploration.
- The first record-stack target should minimize setup risk while staying close enough to the known record path to be a meaningful qMLP control.
- A same-vocab qMLP tax measurement in Phase 14 is only meaningful if this phase reproduces or approximates a strong record-stack control.
- Missing `flash_attn_interface` on A40 is acceptable only when explicitly recorded as an SDPA-fallback screening run.
- The active benchmark sequence is fixed to dense record `sp8192`, qMLP record `sp8192`, and qMLP record `sp16384`; larger vocab frontier work is deferred until those three results exist.

## Implementation Steps

1. Confirm Phase 13 activation.

Before starting jobs, verify that:

- Phase 11 is complete: qMLP `sp16384` lost to replicated `sp8192`.
- Phase 12 is skipped.
- the chosen simple-stack qMLP candidate is explicit: matrix qMLP `sp8192`.
- the plan still calls for record-stack reproduction before more simple-stack exploration.

2. Choose the first record-stack target.

Default selection:

- Start with `2026-04-23_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE` if direct A40 compatibility is uncertain, because it avoids the 04-27 `lrzip` per-group compressor while staying close to the modern CaseOps/SP8192 record path.
- Move to `2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611` only if `flash_attn_interface` and `lrzip` are available or can be handled with bounded effort.
- Keep `2026-04-29_SmearGateBOSFix_3Seed_1.06141` as a compliance/reference target, but treat its small package headroom as a risk for qMLP budget experiments.

3. Inventory the remote environment without running training.

On Slurm compute nodes, use short jobs to record:

- Python version and venv path;
- PyTorch/CUDA versions;
- whether `flash_attn_interface` imports;
- whether Triton imports and can compile a trivial kernel;
- whether `lrzip` is available if the chosen target requires it;
- whether CaseOps prep dependencies are present.

Do not run these diagnostics directly on submit nodes.

Environment check fact as of 2026-06-22:

- job `20484024` ran on A40 and failed because `flash_attn_interface` was missing;
- PyTorch, Triton, SentencePiece, and NumPy imported in the existing venv;
- the failure is now interpreted as requiring the SDPA fallback path for A40 screening, not as a reason to stop Phase 13.

4. Prepare CaseOps data if needed.

If the record target needs `fineweb10B_sp8192_lossless_caps_caseops_v1_reserved`, stage or generate it under `/nfs/hpc/share/peterj29/pg/data-exports/`.

Record:

- data path;
- tokenizer path;
- byte-sidecar path if present;
- shard counts;
- token counts;
- source manifest or prep command.

Use Slurm for any data preparation.

5. Run a minimal record-stack smoke.

Use A40, short walltime, and the chosen record script without qMLP.

The first smoke should test:

- import and initialization;
- one or two training steps;
- validation path if cheap;
- serializer/package path if it can run in short mode;
- memory footprint.

If full package serialization is too expensive for smoke, run the smallest package-size probe that gives a defensible estimate and document the limitation.

6. Run A40 screening benchmarks for a few seeds if smoke passes.

Use the record stack as-is except for hardware-compatible changes that were already documented in smoke.

Use `VOCAB_SIZE=8192`, `QUAT_MLP=0`, and the CaseOps SP8192 tokenizer/data. Prefer seeds matching later qMLP phases, for example `42`, `0`, and `1`, and run independent seeds in parallel when scheduler/account limits allow.

Record for every run:

- job ID;
- seed;
- host and GPU;
- exact command and environment;
- BPB;
- steps;
- `ms/step`;
- package/artifact size;
- memory;
- exit state;
- log paths.

Prefer parallel independent seed jobs when scheduler/account limits allow it.

7. Decide the Phase 14 control.

At the end of this phase, choose one:

- reproduced record stack is strong enough to become the Phase 14 same-vocab qMLP control;
- A40-compatible fallback control is weaker but documented as the best manageable control;
- record-stack reproduction is blocked by FA3/H100-only assumptions and needs a bounded compatibility patch;
- record-stack path is too expensive and should be replaced by a dense near-budget simple-stack control.

## Expected Artifacts

- `goal/13-record.md` updated with the chosen target and results.
- Slurm scripts under `goal/` or a run-specific staging directory.
- Run directories under `/nfs/hpc/share/peterj29/pg/runs/phase13-record-*`.
- Environment logs.
- CaseOps data manifest or prep logs if data prep is needed.
- Smoke logs.
- A40 benchmark logs if smoke passes.

## Data Requirements

The first A40 baseline target, `2026-04-23_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE`, does not use the simple-stack `sp8192` or `sp16384` shards directly. It uses a reserved CaseOps SP8192 tokenizer and a byte sidecar for BPB accounting on original bytes.

Expected tokenizer:

```text
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

Expected dataset directory:

```text
/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
```

Expected shard patterns:

```text
fineweb_train_*.bin
fineweb_val_*.bin
fineweb_val_bytes_*.bin
```

The 04-23 record log reports `train_shards: 80`; the validation path also needs `fineweb_val_bytes_*.bin`, because `CASEOPS_ENABLED=1` scores BPB against original-byte sidecar shards.

Phase look-ahead:

- Phase 13 original record-stack A40 reproduction needs these CaseOps SP8192 shards.
- Phase 14 same-vocab record-stack qMLP tax measurement should reuse the same shards and tokenizer.
- Phase 15 budget-reinvested record-stack qMLP now targets `VOCAB_SIZE=16384` specifically. Larger package-frontier vocab work is deferred until the `sp8192` dense, `sp8192` qMLP, and `sp16384` qMLP A40 comparisons exist.

## Completion Requirements

This phase is complete when:

- a target record stack is chosen with rationale;
- environment compatibility is tested on Slurm compute nodes;
- CaseOps/tokenizer/data prerequisites are either available or the blocker is documented;
- an A40 smoke reaches terminal state, or an import/dependency blocker is captured;
- if smoke passes, at least one A40 record-stack benchmark reaches terminal state;
- all run IDs, commands, logs, package sizes, BPB, step counts, memory, and hardware are recorded;
- `goal/0-plan.md`, this file, and any affected later phase assumptions are updated;
- the Phase 14 same-vocab qMLP-tax target is explicit, or the reason for not proceeding is explicit.

## Failure and Fallback Rules

- If `flash_attn_interface` is missing or H100-only, do not spend days trying to recreate the H100 environment on A40. Either add a bounded A40 attention fallback or choose a lower-ranked compatible control.
- If `lrzip` is unavailable and the chosen target requires `COMPRESSOR=pergroup`, first test `COMPRESSOR=brotli` as a screening fallback and record the artifact-size/BPB implication.
- If CaseOps data prep is slow but progressing normally, let it run as a Slurm job; do not run it on submit nodes.
- If A40 is too slow for full 600-second TTT evaluation, record a no-TTT or reduced-TTT screening metric only as a diagnostic, not as final record-stack proof.
- Do not move to Phase 14 unless this phase leaves a concrete same-vocab control.

## Result

Status: active. CaseOps smoke-data export is proven with the patched prep path; dense record A40 smoke is next while full CaseOps SP8192 data is prepared.

Evidence:

- Read-only inventory found the likely record-stack candidates and their dependency constraints.
- Phase 11 simple-stack `sp16384` completed across three seeds and lost to replicated `sp8192`; Phase 12 was skipped.
- On 2026-06-22, Phase 13 A40 scaffolding was prepared while Phase 11 export jobs were running. The record smoke and benchmark scripts were syntax-checked locally and not submitted; the env check was later submitted as job `20484024`.
- On 2026-06-22, a look-ahead check found that the CaseOps dataset/tokenizer output path was missing, while the source `docs_selected.jsonl`, shipped CaseOps tokenizer, and `prepare_caseops_data.py` were present.
- CaseOps data-prep job `20483645` was submitted and started on `cn-r-1` as `pg-p13-caseops`.
- Early log for `20483645` reached `loaded sp: vocab=8192`, confirming the shipped CaseOps tokenizer loaded.
- Early `sstat` for `20483645.batch` showed `AveCPU=00:02:23` after roughly two minutes elapsed, consistent with the current CaseOps prep script being effectively single-process despite the 16-CPU allocation.
- A later check found `20483645` had run for hours with no output shards. The likely cause was the old validation byte-sidecar path doing repeated prefix decodes before the first shard flush. Do not treat `20483645` as the primary path without rechecking its state.
- Environment check job `20484024` failed because `flash_attn_interface` was missing from the A40 venv.
- The local 04-23 record `train_gpt.py` now includes an SDPA fallback, so the next env/smoke check should allow that fallback and record it as an A40 screening difference.
- The 04-23 `prepare_caseops_data.py` was patched to compute original-byte validation sidecars in one linear pass and to accept `--max-docs`, `--max-train-shards`, and `--dataset-name` so smoke and bounded full exports can use the same path.
- Patched CaseOps smoke-data job `20484885` completed successfully under `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-smoke`, producing one train shard, one validation shard, and one validation-byte sidecar.
- The smoke data is sufficient for a dense record import/package/path smoke, but not for the full Phase 13 A40 baseline. Full CaseOps SP8192 data still needs a replacement patched export or a verified completed existing export.

Artifacts:

- `goal/13-env.sbatch`: A40 compute-node environment and import check for PyTorch/CUDA, Triton, SentencePiece, `flash_attn_interface`, `lrzip`, record files, and CaseOps shard counts.
- `goal/13-caseops-data.sbatch`: CPU Slurm CaseOps data-prep script using the patched 04-23 record `prepare_caseops_data.py`, the shipped CaseOps tokenizer, and an existing `docs_selected.jsonl`.
- `goal/13-smoke.sbatch`: minimal one-GPU A40 record-stack smoke with `PREQUANT_ONLY=1`, `TTT_ENABLED=0`, two iterations, and CaseOps data checks.
- `goal/13-baseline-a40.sbatch`: one-GPU A40 baseline runner for the 04-23 record stack, parameterized by `SEED_VALUE`, with full record settings and default TTT enabled.
- Remote staged scripts include `/nfs/hpc/share/peterj29/pg/runs/phase13-record-env/13-env.sbatch`, `/nfs/hpc/share/peterj29/pg/runs/phase13-record-smoke/13-smoke.sbatch`, and `/nfs/hpc/share/peterj29/pg/runs/phase13-record-baseline/13-baseline-a40.sbatch`. The env check was submitted as job `20484024`; the record smoke and baseline benchmark remain pending data readiness.

New facts:

- The strongest local record candidates are all H100/FA3-oriented and import `flash_attn_interface` directly.
- The 04-27 best record also requires the system `lrzip` binary for its `COMPRESSOR=pergroup` path.
- The 04-23 candidate is slightly weaker but likely simpler as a first A40 reproduction target.
- The 04-23 record's CaseOps data prep is CPU-only and consumes `docs_selected.jsonl`; it can reuse `/nfs/hpc/share/peterj29/pg/data-exports/sp8192-80/docs_selected.jsonl` if that file remains available.
- The 04-23 record's training script expects explicit `DATA_PATH` and `TOKENIZER_PATH` for portable CaseOps runs; the scaffolding writes CaseOps data under `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192/`.
- The 04-23 `prepare_caseops_data.py` remains mostly a single-process Python pipeline, so larger CPU allocations will not necessarily scale linearly. The important speed fix is the linear validation byte-sidecar computation, not more CPUs.
- For the resumed goal, Phase 13 should produce the dense record `sp8192` A40 control. Phase 14 measures qMLP tax at `sp8192`, and Phase 15 tests qMLP reinvestment at `sp16384`.

Decision:

- Resume Phase 13.
- First run the dense record smoke against `/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-smoke` to verify the A40 record path quickly.
- In parallel with that smoke where scheduler/account limits allow, prepare the full CaseOps SP8192 data path with the patched prep script.
- Only cancel old job `20483645` after verifying the patched replacement path is ahead, complete, or otherwise makes the old job wasteful.
- After full CaseOps SP8192 data exists, submit the dense record `sp8192` A40 baseline seeds and keep independent seed jobs parallel where safe.
