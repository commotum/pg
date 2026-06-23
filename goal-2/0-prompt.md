Continue the lean A40 qMLP Parameter Golf work using `goal-2/0-loop.md` and `goal-2/0-plan.md`.

Do not continue the old `goal-1` loop unless explicitly asked. The old loop spent too much A40 time on full phased TTT. The current work is redirected around lean single-GPU A40 screening.

The current decision question is:

```text
Under a lean single-A40 screening setup, does qMLP improve best-under-16MB post-quant no-TTT BPB, and which CaseOps vocab size should be escalated to H100/FA3?
```

Use `goal-2/0-loop.md` as the operating procedure and `goal-2/0-plan.md` as the roadmap. Treat both files as living documents. Follow current evidence, live cluster state, code behavior, benchmark results, and engineering judgment over stale assumptions.

Important constraints:

- Do not run training, tokenizer export, GPU diagnostics, or material compute on submit nodes.
- Use Slurm compute nodes.
- Use CPU Slurm allocations with 16+ CPUs for full CaseOps tokenizer/data exports unless a smaller smoke export is intentional.
- Use A40 single-GPU Slurm jobs for lean smokes and benchmarks.
- Do not request H100/H200, multi-GPU, long jobs, destructive changes, or broad sweeps without explicit approval.
- Parallelize independent jobs when safe, especially seed batches, package smokes, and vocab probes.
- Preserve dependency order: data export -> package/path smoke -> benchmark.
- Do not wait idly for strict phase order when independent useful work is available. It is acceptable to work non-sequentially across phases and matrix cells when dependencies are satisfied.
- If a data export is still running, do other ready work: verify completed exports, patch harnesses, run package smokes for ready vocab/model cells, submit Slurm jobs with `afterok` dependencies, parse completed runs, or update docs.

Lean A40 default:

```text
TTT_ENABLED=0
PHASED_TTT_ENABLED=0
DOCUMENT_PACKING=0 unless proven safe
TORCH_COMPILE=0 unless proven safe
FUSED_MLP_ENABLED=0 unless proven safe
FUSED_CE_ENABLED=0 unless proven safe
attention=SDPA/eager fallback
single GPU
primary score=post-quant no-TTT BPB
```

Use the common record-track CaseOps `sp8192` vocab as the first sanity point, not as the only dense baseline. The benchmark target is a paired dense/qMLP matrix across:

```text
1024
2048
4096
8192
16384
32768
```

For every verified CaseOps vocab size, run both core models:

```text
dense A40-friendly CaseOps
qMLP A40-friendly CaseOps
```

For every valid `(model_variant, vocab_size)` pair, collect three benchmark seeds:

```text
42
0
1
```

Run these as independent one-A40 jobs in parallel where Slurm/account limits allow. If all cells cannot run at once, run the maximum schedulable subset and document the fallback. Use job arrays or a manifest-driven launcher with a concurrency cap when that is cleaner than hand-submitting many commands.

CaseOps policy:

- Do not use simple-stack SentencePiece shards for this goal.
- For each vocab size, use a CaseOps tokenizer trained on `encode_lossless_caps_v2` transformed docs with reserved control symbols and original-byte sidecar accounting.
- Dense and qMLP runs at the same vocab must use the exact same tokenizer model, vocab file, train shards, validation shards, and validation-byte sidecars.
- Different vocab sizes need different CaseOps tokenizer/data exports.
- `sp32768` has been explicitly added to the goal by user direction. It still
  follows the same dependency order: CaseOps export, dense/qMLP smoke, then
  three benchmark seeds for each model. Dense `sp32768` may be over budget; if
  so, keep it labeled as a diagnostic control rather than a compliant
  best-under-cap candidate.

For each phase:

1. Create a detailed phase file in `goal-2/` named `[PHASE-INDEX]-[ONE-WORD-DESCRIPTOR].md`.
2. Implement the phase.
3. Update `goal-2/0-plan.md`, `goal-2/0-loop.md` if needed, the current phase file, and any earlier phase files whose assumptions changed.
4. Decide whether to continue, revise, repeat narrowly, block, abandon, or escalate.

Phase order is a dependency graph, not a blocking queue. If Phase 1 data prep for one vocab is pending but Phase 3/4 work is ready for another vocab, proceed with the ready cells and record the dependency status in the relevant phase files. Do not bypass required gates for a cell: that cell still needs verified data, package/path smoke, then benchmark.

Record for every benchmark:

- job ID;
- command/config;
- git SHAs;
- seed;
- data/tokenizer paths;
- pre-quant BPB;
- post-quant no-TTT BPB;
- total submission bytes;
- quantized model bytes;
- steps;
- ms/step;
- peak GPU memory;
- host/GPU;
- failure mode if any.

Avoid token-wasting loops. If a phase depends on fat work such as full TTT, FA3/Hopper kernels, 8xH100 scaling, or broad full-wallclock tuning, revise the phase into a smaller lean A40 test or ask for explicit approval.
