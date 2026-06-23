# Goal 1 Findings Summary

Date summarized: 2026-06-23

## Scope

This summary covers only the lowest/simple Parameter Golf baseline and the qMLP variants derived from it. It intentionally excludes the later record-stack, CaseOps, TTT, and H100-oriented work.

The main comparison metric is post-quantization int8+zlib roundtrip validation BPB, because that is the value used throughout the phase notes for candidate comparisons. Lower BPB is better.

## Simple-Stack Results

All benchmark rows used the simple 512d / 9 layer stack on A40 with a 600 second training cap unless noted.

### BPB Readings

| Variant | Seed 42 BPB | Seed 0 BPB | Seed 1 BPB | Mean BPB | Notes |
|---|---:|---:|---:|---:|---|
| Dense baseline, `sp1024` | 1.58081095 | n/a | n/a | n/a | One-seed control. |
| qMLP `sp1024`, naive split implementation | 1.87232731 | n/a | n/a | n/a | Correct but too slow and much worse. |
| qMLP `sp1024`, matrix implementation | 1.64863035 | n/a | n/a | n/a | Faster qMLP implementation, still worse than dense. |
| qMLP `sp4096`, matrix | 1.55222627 | 1.54759284 | 1.55872027 | 1.55284646 | First vocab reinvestment win. |
| qMLP `sp8192`, matrix | 1.52530269 | 1.52474158 | 1.52563039 | 1.52522489 | Best simple-stack result. |
| qMLP `sp16384`, matrix | 1.53194348 | 1.53028497 | 1.52768640 | 1.52997162 | Under cap, but worse than `sp8192`. |

### Throughput Readings

| Variant | Seed 42 Steps | Seed 0 Steps | Seed 1 Steps | Mean Steps |
|---|---:|---:|---:|---:|
| Dense baseline, `sp1024` | 379 | n/a | n/a | n/a |
| qMLP `sp1024`, naive split implementation | 263 | n/a | n/a | n/a |
| qMLP `sp1024`, matrix implementation | 368 | n/a | n/a | n/a |
| qMLP `sp4096`, matrix | 352 | 353 | 352 | 352.33 |
| qMLP `sp8192`, matrix | 336 | 337 | 337 | 336.67 |
| qMLP `sp16384`, matrix | 307 | 305 | 308 | 306.67 |

## Conclusions

1. qMLP alone did not help at the baseline `sp1024` vocabulary.

   The naive split qMLP implementation was both slower and much worse than dense. The optimized matrix qMLP implementation fixed most of the speed problem, but still trailed the dense baseline at the same `sp1024` vocabulary: `1.64863035` BPB versus dense `1.58081095`.

2. Vocabulary reinvestment worked up to a point.

   qMLP saved enough parameters to support larger vocabularies. Moving from `sp1024` to `sp4096` and then `sp8192` improved BPB substantially:

   ```text
   qMLP sp1024 matrix:  1.64863035
   qMLP sp4096 mean:    1.55284646
   qMLP sp8192 mean:    1.52522489
   ```

   The larger vocabulary improved tokenization enough to overcome the lower step count through `sp8192`.

3. The benefit saturated at `sp16384`.

   `sp16384` still fit under the artifact cap, but it was slower and produced worse BPB than `sp8192`:

   ```text
   qMLP sp8192 mean:     1.52522489
   qMLP sp16384 mean:   1.52997162
   delta:              +0.00474673 BPB
   ```

   Since lower BPB is better, `sp8192` beat `sp16384`. The likely interpretation is that the extra tokenization benefit from `sp16384` was outweighed by the larger embedding/head/softmax cost and the reduced number of training steps.

4. The best simple-stack qMLP candidate was `sp8192`.

   The simple-stack vocabulary ladder stopped at `sp8192`. `sp16384` did not earn additional replication, and the later work pivoted to dense budget controls and record-stack relevance rather than continuing to larger simple-stack vocabularies.
