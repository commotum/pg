# Phase 7: Vocabulary Reinvestment

Date drafted: 2026-06-22

## Overview

Test the first saved-parameter reinvestment that directly targets BPB: use matrix qMLP's parameter savings to raise the SentencePiece vocabulary from `1024` to `4096`.

Phase 6 showed that matrix qMLP is close enough in A40 throughput to make reinvestment worth testing. It still trails the dense `sp1024` baseline in BPB, so this phase asks whether better tokenization can buy that gap back.

## Current Facts

- Published cached data currently exposes only `sp1024` in `data/manifest.json`.
- The local tokenizer spec only defines `sp_bpe_1024`.
- Larger vocab experiments require rebuilding tokenizer/data from the published `docs_selected.jsonl` cache.
- `sp4096` adds about `(4096 - 1024) * 512 = 1,572,864` tied embedding parameters.
- Matrix qMLP saved `7,077,888` parameters versus dense, so `sp4096` stays well inside the saved-parameter budget.
- Phase 6 provisional matrix qMLP result:

```text
vocab=1024
model_params=9982024
steps=368
step_avg=1630.81ms
roundtrip_val_bpb=1.64863035
```

- Phase 3 dense reference:

```text
vocab=1024
model_params=17059912
steps=379
step_avg=1587.08ms
roundtrip_val_bpb=1.58081095
```

## Implementation Steps

1. Add a local `sp4096` tokenizer config.

Use a config that defines only:

```text
name=sp_bpe_4096
dataset_suffix=sp4096
vocab_size=4096
```

2. Add a bounded data export path.

The exporter now supports:

```text
--max-train-shards 80
```

This should write the full validation split and then stop after 80 full train shards, matching the current benchmark scale without exporting every available published document.

3. Run CPU Slurm data prep, not submit-node preprocessing.

Use:

```text
output_root=/nfs/hpc/share/peterj29/pg/data-exports/sp4096-80
HF_HOME=/nfs/hpc/share/peterj29/pg/hf-cache
MATCHED_FINEWEB_TOKENIZER_THREADS=<cpus>
MATCHED_FINEWEB_SP_BATCH_SIZE=2048
```

The data prep job should:

- download or reuse `docs_selected.jsonl` and sidecar from HF cache;
- train the `sp4096` tokenizer;
- export full validation and 80 train shards;
- write a manifest;
- record output file sizes and shard counts.

4. Validate the export.

Verify:

- tokenizer model exists;
- tokenizer vocab size is `4096`;
- data path contains at least 80 train shards and one validation shard;
- `train_gpt.py` accepts `VOCAB_SIZE=4096` with the new tokenizer path;
- no GPU work happened on submit.

5. Run matrix qMLP `sp4096` smoke.

Use the Phase 6 smoke settings with:

```text
QUAT_MLP=1
QUAT_MLP_IMPL=matrix
VOCAB_SIZE=4096
DATA_PATH=/nfs/hpc/share/peterj29/pg/data-exports/sp4096-80/datasets/fineweb10B_sp4096
TOKENIZER_PATH=/nfs/hpc/share/peterj29/pg/data-exports/sp4096-80/tokenizers/fineweb_4096_bpe.model
```

6. Run matrix qMLP `sp4096` A40 benchmark.

Prefer the exact Phase 3 benchmark resource shape:

```text
partition=share
constraint=a40
gpus=1
cpus-per-task=2
mem=24G
walltime=25m
```

If exact shape is blocked by `QOSGrpCpuLimit`, a 1-CPU provisional run is acceptable, but it must be labeled provisional.

## Completion Requirements

This phase is complete when:

- `sp4096` tokenizer/data export reaches terminal state;
- the export has a recorded manifest, tokenizer vocab size, train shard count, validation shard count, and output path;
- a qMLP matrix `sp4096` smoke reaches `COMPLETED`;
- a qMLP matrix `sp4096` benchmark reaches terminal state;
- `goal/0-plan.md`, this file, and any changed earlier phase assumptions are updated;
- the decision says whether `sp4096` reinvestment improves BPB enough to continue to `sp8192`, width/depth, exact reruns, or stop qMLP.

## Failure and Fallback Rules

- If docs download fails, capture the exact HF/cache error and do not run tokenizer training on submit.
- If full validation plus 80 train shards is too slow, reduce only for a diagnostic smoke and mark it non-comparable.
- If `sp4096` tokenizer training is unstable, retry with fewer tokenizer training docs before changing the model.
- If `sp4096` qMLP is slower and worse BPB than `sp1024` qMLP, stop vocabulary reinvestment.
- If `sp4096` improves BPB but does not beat dense `sp1024`, consider `sp8192` only if artifact/model budget and export cost remain reasonable.
- Do not change width, depth, attention, or quantization in this phase.

## Result

Status: complete as of 2026-06-22 06:41 PDT.

Evidence:

- Added `goal/7-sp4096-tokenizer-config.json` with one tokenizer spec:

```text
name=sp_bpe_4096
dataset_suffix=sp4096
vocab_size=4096
```

- Patched `parameter-golf/data/download_hf_docs_and_tokenize.py` with `--max-train-shards`, so the export can stop after full validation plus 80 full train shards instead of exporting the whole docs snapshot.
- Added CPU Slurm scripts:
  - `goal/7-docs.sbatch` for selected-docs materialization;
  - `goal/7-data.sbatch` for bounded `sp4096` tokenizer/data export.
- The first `sp4096` export attempts were staged but canceled while pending:
  - job `20480667`, 4 CPUs / 32G, canceled on `QOSGrpCpuLimit`;
  - job `20480671`, 2 CPUs / 24G, canceled on `QOSGrpCpuLimit`.
- Docs materialization job `20480690` ran on `cn-a26`, partition `share`, 1 CPU, 16G RAM, and completed with state `COMPLETED`, exit code `0:0`, elapsed `00:08:55`.
- Docs materialization command:

```text
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0 --with-docs
```

- Materialized docs:

```text
data/docs_selected.jsonl: 45G
data/docs_selected.source_manifest.json: 481 bytes
docs_sha256: 84386dfa7b339a5d4831d5273c4a2028b78b60670d3a235633a8520545d19bc7
sidecar_sha256: 2db506dcd4cbcaabac767bf766a42a6e2b99fed28055c6dd48bbd9f88d491ab0
HF cache size after materialization: 60G
```

- Sidecar facts:

```text
snapshot_kind=partial_docs_cache_from_50B_export
selection_seed=1337
num_docs=15368808
docs_val=50000
docs_train=15318808
docs_bytes=48166275520
docs_sha256=84386dfa7b339a5d4831d5273c4a2028b78b60670d3a235633a8520545d19bc7
```
- Bounded `sp4096` export job `20480717` ran on `cn-a14`, partition `share`, 2 CPUs, 24G RAM, and completed with state `COMPLETED`, exit code `0:0`, elapsed `02:10:25`.
- `sp4096` export command:

```text
python data/download_hf_docs_and_tokenize.py \
  --output-root /nfs/hpc/share/peterj29/pg/data-exports/sp4096-80 \
  --tokenizer-config /nfs/hpc/share/peterj29/pg/runs/phase7-sp4096-data/sp4096-tokenizer-config.json \
  --tokenizer-train-docs 500000 \
  --max-train-shards 80 \
  --skip-byte
```

- `sp4096` export verification:

```text
manifest=/nfs/hpc/share/peterj29/pg/data-exports/sp4096-80/manifest.json
tokenizer=sp_bpe_4096
sp_vocab_size=4096
dataset=fineweb10B_sp4096
train_files=80
val_files=1
train_tokens=8000000433
val_tokens=45517764
tokenizer_model=302569 bytes
tokenizer_vocab=51568 bytes
```

- Matrix qMLP `sp4096` smoke job `20480883` ran on `cn-r-5`, partition `share`, one `NVIDIA A40`, 1 CPU, 16G RAM, and completed with state `COMPLETED`, exit code `0:0`, elapsed `00:04:13`.
- `sp4096` smoke facts:

```text
model_params:11554888
step:2/2 val_loss:8.3065 val_bpb:3.6105 train_time:639ms step_avg:319.65ms
peak memory allocated: 1808 MiB reserved: 1936 MiB
Serialized model int8+zlib: 6403349 bytes
Total submission size int8+zlib: 6454558 bytes
final_int8_zlib_roundtrip_exact val_loss:8.30869535 val_bpb:3.61143104
```

- Matrix qMLP `sp4096` benchmark job `20480898` ran on `cn-r-3`, partition `share`, one `NVIDIA A40`, 2 CPUs, 24G RAM, and completed with state `COMPLETED`, exit code `0:0`, elapsed `00:14:25`.
- `sp4096` benchmark facts:

```text
model_params:11554888
train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:600.000
quat_mlp:True
quat_mlp_impl:matrix
step:352/20000 val_loss:3.5590 val_bpb:1.5470 train_time:601019ms step_avg:1707.44ms
stopping_early: wallclock_cap train_time:601019ms step:352/20000
peak memory allocated: 13443 MiB reserved: 13454 MiB
Serialized model: 42076091 bytes
Total submission size: 42127300 bytes
Serialized model int8+zlib: 9931222 bytes
Total submission size int8+zlib: 9982431 bytes
final_int8_zlib_roundtrip_exact val_loss:3.57115367 val_bpb:1.55222627
```

- Comparison against the main references:

```text
dense_sp1024_phase3_roundtrip_val_bpb=1.58081095
qmlp_matrix_sp1024_phase6_provisional_roundtrip_val_bpb=1.64863035
qmlp_matrix_sp4096_phase7_roundtrip_val_bpb=1.55222627

dense_sp1024_phase3_steps=379
qmlp_matrix_sp1024_phase6_provisional_steps=368
qmlp_matrix_sp4096_phase7_steps=352

dense_sp1024_phase3_step_avg=1587.08ms
qmlp_matrix_sp4096_phase7_step_avg=1707.44ms

dense_sp1024_phase3_model_params=17059912
qmlp_matrix_sp4096_phase7_model_params=11554888
```

- Deltas:

```text
sp4096_qmlp_vs_dense_roundtrip_bpb_delta=-0.02858468
sp4096_qmlp_vs_sp1024_qmlp_roundtrip_bpb_delta=-0.09640408
sp4096_qmlp_vs_dense_steps_delta=-27
sp4096_qmlp_vs_dense_model_params_delta=-5505024
```

Artifacts:

- Local phase plan: `goal/7-vocab.md`.
- Local tokenizer config: `goal/7-sp4096-tokenizer-config.json`.
- Local docs job: `goal/7-docs.sbatch`.
- Local data export job: `goal/7-data.sbatch`.
- Remote docs artifacts: `/nfs/hpc/share/peterj29/pg/runs/phase7-docs/20480690/`.
- Remote materialized docs:
  - `/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf/data/docs_selected.jsonl`;
  - `/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf/data/docs_selected.source_manifest.json`.
- Remote `sp4096` export root: `/nfs/hpc/share/peterj29/pg/data-exports/sp4096-80/`.
- Remote `sp4096` data artifacts: `/nfs/hpc/share/peterj29/pg/runs/phase7-sp4096-data/20480717/`.
- Remote `sp4096` smoke artifacts: `/nfs/hpc/share/peterj29/pg/runs/phase7-sp4096-smoke/20480883/`.
- Remote `sp4096` benchmark artifacts: `/nfs/hpc/share/peterj29/pg/runs/phase7-sp4096-benchmark/20480898/`.

New facts:

- The selected docs cache is large but already materialized on HPC, so the next export job should reuse local/HF-cached files rather than downloading 45G again.
- The docs sidecar says this is a partial docs cache from a 50B export, not a canonical 10B shard selection. That matches the published cached baseline source and is still the right source for matched retokenization.
- `sp4096` export is feasible on a small 2-CPU CPU node, but it takes about 2 hours for tokenizer training plus full validation and 80 train shards.
- `sp4096` qMLP matrix is slower than dense `sp1024`, but its tokenizer improvement more than compensates under the same 600-second cap.
- This is the first positive answer to the core question: qMLP's saved parameters can be reinvested into vocabulary to improve validation BPB.

Decision:

- Proceed to seed replication before `sp8192` or width/depth.
- Use the exact Phase 7 `sp4096` benchmark shape as the candidate to replicate.
- If at least one additional seed stays ahead of dense `sp1024`, then consider `sp8192` or a stronger record-stack integration.
