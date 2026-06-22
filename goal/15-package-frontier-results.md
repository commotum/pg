# Phase 15 Results: qMLP Record-Stack Package Frontier

Date: 2026-06-22

This note records the current best facts for the Phase 15 record-stack qMLP package-size search. It should be read alongside `goal/15-vocab-max.md`; where this note conflicts with that draft estimate, this note is newer.

## Executive Summary

The first record-stack qMLP package canary at `VOCAB_SIZE=11776` completed successfully and produced a much smaller package than the original Phase 15 estimate implied. The old `11k-12k` expected frontier was too conservative.

The canary does not prove final trained BPB or final trained package size. It does prove that the record-stack qMLP packaging path is mechanically working and that `11776` is far below the 16,000,000 byte cap in the current smoke setup.

Current implication: do not spend time on small vocab increments around 12k. `16384` is likely safe enough to use as a sanity checkpoint, but it is probably not the real package frontier. The next useful Phase 15 probe should jump to a serious bracket such as `24576`, then use a high bracket such as `32768` if the package remains safely under cap.

## Simple-Stack Context

The simple-stack qMLP vocabulary ladder established that qMLP can convert saved MLP budget into useful vocabulary:

| Candidate | Result |
| --- | --- |
| qMLP `sp4096` | Three-seed mean roundtrip BPB `1.55284646`; about `0.0280` BPB better than dense `sp1024`. |
| qMLP `sp8192` | Three-seed mean roundtrip BPB `1.52522489`; about `0.0276` BPB better than replicated `sp4096`. |
| qMLP `sp16384` | Three-seed mean roundtrip BPB `1.52997162`; about `0.00474673` BPB worse than replicated `sp8192`. |

Simple-stack conclusion: `sp8192` is the best simple-stack qMLP point seen so far. Simple-stack vocab expansion stopped because `sp16384` fit under the package cap but did not improve BPB.

That does not answer the main question anymore. The current question is whether qMLP can beat or plausibly match a relevant record-stack control after reinvesting its budget.

## Phase 15 Canary Result

Successful package smoke:

```text
job_id: 20484777
state: COMPLETED
exit_code: 0:0
elapsed: 00:08:35
node: cn-r-5
vocab_size: 11776
quat_mlp: true
quat_mlp_impl: matrix
model_params: 20479162
quantized_model_brotli_bytes: 9344417
total_submission_bytes: 9376232
cap_bytes: 16000000
headroom_bytes: 6623768
headroom_mib: 6.3169
```

Log path:

```text
/nfs/hpc/share/peterj29/pg/runs/phase15-qmlp-package-smoke-fitdummy7/20484777/train.log
```

The package used the record-stack qMLP path with dummy shards and A40-safe compatibility switches. The compressed submission was `9,376,232` bytes, which is about `9.38 MB` decimal or `8.94 MiB`.

## Packaging Details Observed

The successful canary reached the GPTQ/Brotli packaging path:

```text
Serialized model: 69880891 bytes
Code size (uncompressed): 154501 bytes
Code size (compressed): 31815 bytes
GPTQ: collected 133 Hessians in 1.3s
```

Observed quantization categories included:

```text
gate_int8_row: blocks.attn.attn_gate_w
gptq int6: attention weights and qMLP quaternion components
gptq int7: tok_emb.weight
passthrough: small tensors
```

This is important because it means the result is not just a raw parameter-count estimate. The qMLP record-stack package path is exercising the optimized matrix qMLP implementation and the real compressed export path.

## Interpretation

The `11776` result changes the Phase 15 search strategy:

- The old `11k-12k` frontier estimate in `goal/15-vocab-max.md` is superseded.
- `11776` is a lower bound, not a near-frontier result.
- `16384` is likely to fit mechanically, but it should not be treated as the likely maximum.
- The useful work is now to bracket the true frontier with large jumps, then binary-search with 512-token granularity.

A rough conservative storage estimate from `11776` to `16384`:

```text
additional_tokens = 4608
tied_embedding_width = 512
int8_like_raw_bytes = 4608 * 512 = 2359296
int7_raw_payload_bytes = 2064384
```

Adding this to the canary package gives an expected `16384` smoke package around `11.5-12.0 MB` before trained-weight entropy and other overhead. That is why `16384` looks safe, but also why it is probably not the final frontier.

## Caveats

This was a package canary, not a full benchmark.

- The run used dummy shards, so its BPB is not meaningful.
- Tiny package smokes can underpredict full trained package size because trained weights are usually less compressible than near-initial or dummy-trained weights.
- A candidate that fits in this smoke still needs a trained package check before being considered competition-safe.
- A40 compatibility switches were active: `DOCUMENT_PACKING=0`, `TORCH_COMPILE=0`, `FUSED_MLP_ENABLED=0`, plus SDPA fallback when `flash_attn_interface` is unavailable.
- Those switches are appropriate for a sizing/mechanical smoke, but they are not the final H100/FA3 performance path.

## Issues Already Cleared

The successful canary came after several mechanical blockers were fixed or routed around:

- missing `flash_attn_interface` on A40 path;
- packed-document varlen fallback hitting TorchDynamo compile guards;
- `torch.compile` tracing/Triton issues with the fallback/qMLP path;
- fused Triton MLP exceeding A40 shared memory;
- missing `pyminify`;
- missing `brotli`.

The current smoke recipe avoids those blockers for package-size probing.

## Recommended Next Probes

Use the existing Phase 15 scripts:

```text
VOCAB_SIZE=<candidate> sbatch goal/15-caseops-vocab.sbatch
VOCAB_SIZE=<candidate> sbatch goal/15-qmlp-package-smoke.sbatch
```

Recommended bracket:

1. Probe `VOCAB_SIZE=24576` next as a serious lower-bound candidate.
2. If `24576` is comfortably under cap, probe `VOCAB_SIZE=32768` as an aggressive upper bracket.
3. If `32768` fails or is too close to cap, binary search between `24576` and `32768`.
4. If `24576` is already too close or over cap, binary search between `11776` and `24576`.
5. Once a near-frontier smoke candidate exists, run the full A40 record-stack qMLP benchmark only for the best safe candidate or one small pair of serious candidates.

For larger vocab candidates, prefer enough tokenizer training documents to avoid a weak tokenizer. `TOKENIZER_TRAIN_DOCS=100000` is safer than `50000` for very large vocab probes if queue time allows.

## Current Decision State

Phase 15 remains active. The current best factual lower bound is:

```text
record-stack qMLP VOCAB_SIZE=11776 fits at 9376232 total submission bytes
```

The next decision should not be whether `16384` fits. The next decision should be how high the record-stack qMLP vocabulary can go while leaving enough headroom for a trained package and then whether that best-under-budget candidate can beat the record-stack A40 baseline.
