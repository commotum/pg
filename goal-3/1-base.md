# Phase 1: Base

## Overview

Select the exact H100 record stack to modify, identify the fallback, and map the
code surfaces that qMLP must touch before implementation. This phase does not
modify training code or submit jobs.

## Selected Base

Primary base:

```text
parameter-golf/records/track_10min_16mb/
2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611
```

Use this as the base for Goal 3 because it is the strongest local full H100
record candidate:

- reported 3-seed post-TTT mean: `1.06107587` BPB;
- reported std: `0.00090` BPB;
- reported max artifact: `15,907,550` bytes;
- reported mean artifact: `15,901,919` bytes;
- reported mean training steps: `4931.33`;
- reported mean step time: `121.7 ms`;
- hardware: `8xH100 80GB SXM`;
- stack: CaseOps `sp8192`, FA3, XSA, SparseAttnGate, BOS-fixed SmearGate,
  Polar-Express Muon, fused CE, GPTQ int6/int7, LQER asymmetric rank-4,
  per-group `lrzip` compression, and phased score-first TTT.

Fallback:

```text
parameter-golf/records/track_10min_16mb/
2026-04-29_SmearGateBOSFix_3Seed_1.06141
```

Use this only if the primary base cannot be staged or made compliant on OSU.
It is slightly weaker but has clearer compliance timing around
`GPTQ_RESERVE_SECONDS=8.0`:

- original 3-seed mean: `1.06145` BPB;
- compliance rerun mean: `1.06141` BPB;
- max artifact in reproduction: `15,952,690` bytes;
- rerun training plus GPTQ Hessian collection reported under `596s`.

## Primary Reproduction Command

The primary README's reproduction command is the starting command shape:

```bash
DATA_DIR=./data \
VOCAB_SIZE=8192 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Goal 3 will add controlled qMLP flags on top of this shape, not change unrelated
settings first.

## Required Environment

From the primary base:

- Python dependencies from `requirements.txt`:
  - `torch==2.9.1+cu128`;
  - `sentencepiece`;
  - `brotli`;
  - `huggingface_hub`;
  - `numpy`;
  - `python-minifier`.
- FlashAttention 3 installed separately:

```bash
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

- CUDA target: `12.8`.
- System binary: `lrzip`, required for `COMPRESSOR=pergroup`.
- H100 target: one node with 8 H100 80GB GPUs. On OSU, do not use
  `--constraint=h100` alone; Phase 0 showed it can target `h100-40g`.

## qMLP Source Pattern

Use the banked qMLP implementation from:

```text
parameter-golf/records/track_10min_16mb/
2026-04-23_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE/train_gpt.py
```

This is a better port source than the simple `QuaternionLinear` in
`parameter-golf/train_gpt.py` because it already matches the record-stack
banked MLP layout.

Relevant 04-23 qMLP surfaces:

- Hyperparameters:
  - `QUAT_MLP`;
  - `QUAT_MLP_IMPL`, matrix only.
- Helpers:
  - `_quaternion_matrix(w)`;
  - `_quaternion_input_hessian(x)`;
  - `_QUAT_COMPONENTS`;
  - `_is_quat_mlp_component(name)`.
- MLP forward:
  - detects 3D quaternion component bank tensors;
  - materializes a Hamilton dense matrix through `_quaternion_matrix`;
  - reuses the fused LeakyReLU-square MLP path after materialization.
- GPT parameter banks:
  - dense: `(num_layers, hidden_dim, model_dim)` and
    `(num_layers, model_dim, hidden_dim)`;
  - qMLP: `(num_layers, 4, hidden_dim // 4, model_dim // 4)` and
    `(num_layers, 4, model_dim // 4, hidden_dim // 4)`.
- Initialization:
  - initializes each quaternion component matrix;
  - zero-inits qMLP down bank.
- GPTQ:
  - qMLP-specific Hessian construction with `_quaternion_input_hessian`;
  - qMLP component names forced through GPTQ even when small;
  - unbank/rebank support for `blocks.{i}.mlp.fc.{wr,wi,wj,wk}` and
    `blocks.{i}.mlp.proj.{wr,wi,wj,wk}`.

## Primary 04-27 Code Surfaces

These are the primary-base locations to map during implementation:

| Surface | Primary-base location |
|---|---|
| Hyperparameters | `train_gpt.py:227` |
| MLP class | `train_gpt.py:1072` |
| GPT parameter banks | `train_gpt.py:1154-1158` |
| MLP bank initialization | `train_gpt.py:1266-1268` |
| `_bank_weights` | `train_gpt.py:1280` |
| normal forward hidden path | `train_gpt.py:1324` |
| `forward_ttt` | `train_gpt.py:1427` |
| TTT MLP LoRA hooks | `train_gpt.py:1561-1565`, `train_gpt.py:1621-1626` |
| `BatchedTTTLoRA` | `train_gpt.py:1663` |
| Muon/Adam optimizer grouping | `train_gpt.py:1900` |
| fp32 restore | `train_gpt.py:2022` |
| GPTQ Hessian collection | `train_gpt.py:2039` |
| mixed GPTQ quantization | `train_gpt.py:2220` |
| per-group compression | `train_gpt.py:2352` |
| unbank/rebank | `train_gpt.py:2573`, `train_gpt.py:2598` |
| serialization/GPTQ | `train_gpt.py:2645` |
| phased TTT eval | `train_gpt.py:3021` |
| training loop and wallclock cap | `train_gpt.py:3331` |
| train/eval/package orchestration | `train_gpt.py:3553` |

## qMLP Insertion Plan

Add qMLP to the primary base with the smallest useful change:

1. Add `QUAT_MLP` and `QUAT_MLP_IMPL` hyperparameters.
2. Add qMLP helpers from the 04-23 record.
3. Teach `GPT.__init__` to allocate dense or qMLP MLP banks.
4. Preserve the dense path when `QUAT_MLP=0`.
5. Teach `MLP.forward` to materialize qMLP banks through `_quaternion_matrix`.
6. Preserve the existing fused MLP path after qMLP matrix materialization.
7. Update initialization for qMLP component banks.
8. Keep qMLP banks in Muon matrix params.
9. Update GPTQ Hessian collection for qMLP components.
10. Force qMLP components through GPTQ rather than float16 passthrough.
11. Update `_unbank_state_dict` and `_rebank_state_dict`.
12. Update per-group compression for qMLP component key names.
13. Verify `forward_ttt`, `_block_with_lora`, and `_parallel_block_with_lora`
    use exactly the same qMLP MLP path as normal eval/training.

## Compliance-Sensitive Risks

1. **H100 target constraint.** `--constraint=h100` can target a 40GB H100 node.
   Final scripts must use the live-validated H100 80GB feature expression.
2. **Package headroom.** The primary base already reaches about `15.9 MB`.
   qMLP should save model bytes, but code changes, vocab changes, and per-group
   compression behavior must be measured. No H100 final run should proceed
   without package smokes.
3. **Per-group compression.** The 04-27 base adds per-group `lrzip`
   compression. The 04-23 qMLP source does not appear to cover the newer
   per-group key buckets for qMLP component names. This is a required Goal 3
   implementation surface.
4. **GPTQ Hessians.** qMLP component tensors need matching Hessians. If this is
   missed, quantization either fails or silently treats qMLP tensors incorrectly.
5. **TTT MLP LoRA.** Existing TTT MLP LoRA is an additive `dim -> dim` path on
   the MLP input/output space, not a replacement for the hidden projection. It
   can likely remain dimensionally unchanged, but it must be verified on both
   `_block_with_lora` and `_parallel_block_with_lora`.
6. **Fused MLP speed.** The qMLP matrix path materializes dense Hamilton
   matrices before calling the fused MLP. This preserves the fused kernel but
   may add compile/runtime overhead. H100 smoke must record throughput before
   committing to a final candidate.
7. **Training budget compliance.** The 04-27 base uses
   `GPTQ_RESERVE_SECONDS=0.5`; the 04-29 compliance rerun suggests `8.0` is
   safer for GPTQ Hessian timing. Goal 3 must decide this explicitly before
   final H100 submission.
8. **CaseOps sp16384 compatibility.** The primary record ships `sp8192`. The
   `sp16384` candidate needs compatible CaseOps data/tokenizer/byte sidecars and
   must preserve original-byte BPB accounting.

## Verification

Phase 1 verification is documentation-only:

- primary and fallback READMEs inspected;
- primary `requirements.txt` and `submission.json` inspected;
- fallback `submission.json` inspected;
- primary `train_gpt.py` surfaces scanned;
- 04-23 qMLP source surfaces scanned;
- no compute jobs submitted.

## Completion Requirements

- `goal-3/1-base.md` exists: complete.
- selected base is justified: complete.
- fallback base is recorded: complete.
- qMLP insertion points are identified: complete.
- compliance risks are listed before implementation: complete.

## Next Phase

Phase 2: verify or prepare compatible CaseOps `sp8192` and `sp16384`
tokenizer/data manifests before implementing H100 runners.
