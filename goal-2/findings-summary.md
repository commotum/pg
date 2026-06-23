# Goal 2 Findings Summary

Date summarized: 2026-06-23

## Scope

Goal 2 is the A40-friendly record-track exploration: CaseOps-style vocabulary/data, no TTT, no H100-only FA3 path, and no scarce H100/H200 spend unless A40 evidence earns it. These BPB numbers should be compared within Goal 2 only. They are not directly comparable to Goal 1's simple-stack BPB scale because the stack, tokenizer/data path, and benchmark harness changed.

The core question was whether a lean single-A40 record-track setup can show qMLP as useful under the 16 MB package cap, and whether qMLP remains useful after dense controls are given a fair vocab budget.

## Main Result

qMLP is much stronger in Goal 2 than it was in Goal 1. In the A40-friendly CaseOps record-track setup, qMLP beats dense at every matched vocab size we tested, and the best qMLP run uses the package savings to move to `sp16384`.

Current best clean under-cap candidates:

| Model | Vocab | Mean Quantized BPB | Package Status | Role |
|---|---:|---:|---|---|
| dense | 8192 | 3.66039918 | under 16 MB | best clean dense control |
| qMLP | 16384 | 2.99609830 | under 16 MB | best clean qMLP candidate |
| qMLP | 32768 | 3.03410127 | under 16 MB | diagnostic frontier, worse than `sp16384` |

Lower BPB is better. The best qMLP candidate beats the best clean dense candidate by about `0.66430088` BPB on the A40 screening harness.

## Clean Matrix

These are three-seed means unless noted in the phase docs. The dense `sp16384` and dense `sp32768` rows are excluded here because they exceeded the 16 MB package cap.

| Model | Vocab | Mean Quantized BPB | Notes |
|---|---:|---:|---|
| dense | 1024 | 4.62124177 | clean under-cap matched baseline |
| qMLP | 1024 | 3.36463320 | clean under-cap matched qMLP |
| dense | 2048 | 4.26447164 | clean under-cap matched baseline |
| qMLP | 2048 | 3.25738655 | clean under-cap matched qMLP |
| dense | 4096 | 3.94311933 | clean under-cap matched baseline |
| qMLP | 4096 | 3.11496067 | clean under-cap matched qMLP |
| dense | 8192 | 3.66039918 | best clean dense control |
| qMLP | 8192 | 3.01745760 | strong qMLP, but below `sp16384` |
| qMLP | 16384 | 2.99609830 | best clean qMLP result |
| qMLP | 32768 | 3.03410127 | clean under-cap, but worse than `sp16384` |

Matched-vocab qMLP wins:

| Vocab | Dense Mean BPB | qMLP Mean BPB | qMLP Gain |
|---:|---:|---:|---:|
| 1024 | 4.62124177 | 3.36463320 | 1.25660857 |
| 2048 | 4.26447164 | 3.25738655 | 1.00708509 |
| 4096 | 3.94311933 | 3.11496067 | 0.82815866 |
| 8192 | 3.66039918 | 3.01745760 | 0.64294158 |

The qMLP gain shrinks as vocab grows, but it remains large through `sp8192`. The dominant effect is the matched-vocab qMLP gain; larger vocab then adds a smaller improvement on top, with `sp16384` beating `sp8192` by about `0.02135930` BPB.

## Over-Budget Dense Diagnostics

The dense high-vocab runs are useful diagnostics, but they are not compliant candidates under the 16 MB package limit.

| Model | Vocab | Mean Quantized BPB | Package Status | Interpretation |
|---|---:|---:|---|---|
| dense | 16384 | 3.57978517 | over 16 MB | diagnostic only |
| dense | 32768 | 3.56331962 | over 16 MB | diagnostic only |

Dense does improve when pushed beyond `sp8192`, but the resulting submissions are over budget. This supports the narrower view that additional vocabulary can improve scores. In this harness, qMLP's larger contribution is the standalone matched-vocab gain; vocabulary augmentation adds a smaller additional benefit on top of that.

## `sp32768` Cleanup

The original `sp32768` 75-minute jobs reached final diagnostics but some timed out before writing `COMPLETE.txt`. We reran the missing clean completions with a 3-hour reservation and cancelled none manually; all three reruns completed naturally in about 25-32 minutes.

Clean `sp32768` rows now used for findings:

| Model | Seed | Job | Quantized BPB | Total Submission Bytes | Steps | Status |
|---|---:|---:|---:|---:|---:|---|
| dense | 42 | 20487148 | 3.58922764 | 22498076 | 57 | completed |
| dense | 0 | 20486315 | 3.55337067 | 22491882 | 57 | completed |
| dense | 1 | 20486316 | 3.54736056 | 22499431 | 57 | completed |
| qMLP | 42 | 20487146 | 3.04764432 | 15021645 | 57 | completed |
| qMLP | 0 | 20487147 | 3.03884955 | 15022871 | 57 | completed |
| qMLP | 1 | 20486319 | 3.01680995 | 15023513 | 57 | completed |

Clean means:

| Model | Vocab | Mean Quantized BPB | Mean Prequant BPB | Mean Steps |
|---|---:|---:|---:|---:|
| dense | 32768 | 3.56331962 | 3.55820177 | 57 |
| qMLP | 32768 | 3.03410127 | 3.03255332 | 57 |

The clean `sp32768` qMLP result is under the cap, but it is worse than `sp16384` qMLP by about `0.03800297` BPB. That makes `sp16384` the better carry-forward qMLP candidate.

## Difference From Goal 1

Goal 1 found that qMLP was not useful at the same small vocabulary in the simple stack, but became useful when the parameter savings were reinvested into vocab growth. The best simple-stack qMLP setting was `sp8192`; `sp16384` was under budget but slightly worse.

Goal 2 is different:

- qMLP wins at the same vocab size in the A40-friendly CaseOps record-track setup.
- qMLP's best vocab moved from `sp8192` in Goal 1 to `sp16384` in Goal 2.
- `sp32768` remains under budget for qMLP, but appears past the useful vocab frontier for this training budget.
- Dense high-vocab controls improve BPB, but exceed the 16 MB package cap, so they do not displace the compliant qMLP result.
- The A40 record-track runs are heavier per step than Goal 1 simple-stack runs; large-vocab record-track jobs typically reached around 57-66 benchmark steps rather than hundreds of simple-stack steps.

The most important practical shift is that Goal 2 no longer frames qMLP as "same model, worse architecture, recovered by bigger vocab." In the CaseOps record-track harness, qMLP is directly better than dense at matched vocab and also allows larger useful vocab under budget.

## Current Conclusions

1. The best clean A40-friendly qMLP candidate is `qMLP sp16384`, with mean quantized BPB `2.99609830`.
2. The best clean dense control is `dense sp8192`, with mean quantized BPB `3.66039918`.
3. qMLP `sp32768` is valuable as a frontier probe, but it should not replace `sp16384`.
4. Dense `sp16384` and `sp32768` are useful diagnostics, but they are over-budget and should be excluded from compliant rankings.
5. The A40 evidence is strong enough to preserve `qMLP sp16384` as the carry-forward candidate, but not by itself a reason to launch scarce H100/H200 jobs automatically.
6. If confirmation is pursued later, it should start from the exact `qMLP sp16384` record-track configuration, with the command/config reviewed before using H100/H200 resources.

## Caveats

- This is A40 screening evidence, not a final H100/FA3 record attempt.
- The no-TTT constraint was intentional for lean single-GPU exploration.
- Absolute BPB values should not be compared directly to Goal 1 simple-stack results.
- The `sp32768` reruns are included here, but other phase tables may still need doc cleanup if they were written before the 3-hour reruns completed.
