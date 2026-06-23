# Goal 3 Findings Summary

Date started: 2026-06-23

## Scope

Goal 3 is the full 8xH100 H100/FA3 record-track qMLP attempt. It carries
forward Goal 2's A40 evidence, but Goal 3 results must be judged only after the
full record stack runs on the intended H100 class.

## Current Findings

1. The primary base should remain the 2026-04-27 record candidate unless Phase 1
   code inspection finds a blocker. It is the strongest local H100 record note,
   with reported 3-seed post-TTT mean `1.06108` BPB and about `15.9 MB`
   artifacts.
2. The 2026-04-29 reproduction is the fallback because it has clearer compliance
   timing around `GPTQ_RESERVE_SECONDS=8.0` and reports post-TTT mean `1.06141`
   BPB.
3. Goal 2's carry-forward qMLP candidate is `sp16384`; `sp32768` is under cap
   in the A40 harness but worse than `sp16384`.
4. Live OSU Slurm checks showed `--constraint=h100` alone is too broad for the
   intended run because it can select `dgxh-1` with `gpu:h100-40g:16`.
   Goal 3 H100 scripts must use the current live-validated 80GB constraint,
   `--constraint="h100&vram80g"`, unless a later live check changes the correct
   feature expression.
5. The remote HPC checkout exists but is stale relative to the local Goal 3
   files. It must be synced before Slurm work.
6. Phase 1 selected the 2026-04-23 record-stack qMLP implementation as the best
   port source because it already handles banked MLP weights, qMLP component
   Hessians, forced GPTQ of quaternion components, and unbank/rebank logic.
7. The 2026-04-27 primary base adds per-group `lrzip` compression that the
   04-23 qMLP source does not fully cover. qMLP component names must be wired
   into compression/package handling before package smokes.
8. The 04-27 primary base used `GPTQ_RESERVE_SECONDS=0.5`; the 04-29 compliance
   reproduction showed `GPTQ_RESERVE_SECONDS=8.0` keeps training-data-access
   timing safer with negligible BPB change in that stack. Goal 3 must choose
   this deliberately before final H100 execution.
9. Phase 2 found both required CaseOps exports already present on HPC shared
   storage: `caseops-sp8192-patched` and `caseops-sp16384`. Each has 80 train
   shards plus validation token and validation-byte sidecar files.
10. The remote `sp8192` tokenizer hash matches the local tokenizer shipped with
    both the 2026-04-27 primary base and the 2026-04-29 fallback.
11. Phase 3 staged a qMLP-enabled copy of the 2026-04-27 record stack under
    `goal-3/stage/primary-qmlp/`. The dense path remains selected by
    `QUAT_MLP=0`; qMLP is selected by `QUAT_MLP=1`.
12. The staged qMLP port adds compact qMLP MLP banks, Hamilton matrix
    materialization, qMLP Hessians, forced GPTQ for qMLP components,
    qMLP-aware unbank/rebank, and qMLP component keys for per-group
    compression.
13. Local static checks pass for the staged training file and all current Goal 3
    helper/sbatch scripts.
14. The default one-hour H100 runner order is now
    `dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384`. This prioritizes the
    `sp16384` qMLP record contender after minimal dense/qMLP smoke validation.
15. The H100 runner now records dirty Git evidence through `git-status.txt`,
    `git-diff.stat`, and `git-diff.patch`, which is required for later
    compliance review.
16. The H100 short-smoke and record-runner scripts now stage the Goal 3 source
    tree and both required CaseOps data/tokenizer sets to node-local scratch
    under `/scratch/$USER/$SLURM_JOB_ID/goal3`.
17. `goal-3/scripts/env_smoke.py` centralizes the H100 environment smoke:
    8 visible CUDA devices, FA3 import, `lrzip`, and `sp8192`/`sp16384`
    tokenizer vocab sizes.
18. `goal-3/compliance-note.md` now documents the compliance assumptions and
    explicitly does not claim final compliance until runtime package/BPB checks
    pass.
19. `goal-3/` has been synced to the remote HPC checkout at
    `/nfs/hpc/share/peterj29/pg/src/pg/goal-3`. Remote submit-node static checks
    pass for the Goal 3 shell scripts and Python syntax.
20. CPU environment prep completed as Slurm job `20487397`, producing
    `/nfs/hpc/share/peterj29/pg/envs/goal3-cu128` with Python 3.12,
    `torch==2.9.1+cu128`, Triton, `sentencepiece`, `brotli`, and
    `flash_attn_3`.
21. CPU tools prep completed as Slurm job `20487617`, producing user-local
    `lrzip 0.651` under `/nfs/hpc/share/peterj29/pg/tools/lrzip/bin/lrzip`.
    The build required user-local LZO and LZ4. The binary cannot be validated by
    executing it on the submit node because the submit node has an older glibc;
    the valid next check is inside the H100 env smoke allocation.
22. Phase 7 live Slurm refresh on 2026-06-23 at 16:05 Pacific still shows
    `dgxh-3` as the valid H100 80GB target class for
    `--constraint="h100&vram80g"`. `dgxh-1` remains unsuitable for the intended
    competition-class run because it is advertised as `h100-40g`.
23. `srun --test-only` for both the 15-minute H100 env smoke and the one-hour
    record runner currently predicts `dgxh-3` at `2026-06-27T08:29:30`.
24. The H100 approval packet now exists at `goal-3/7-approval.md`. It requests
    approval only for `goal-3/h100-env-smoke.sbatch`; approval for the later
    one-hour record runner remains separate and unrequested.
25. Static pre-H100 audit found that `env_smoke.py` previously only checked
    whether `lrzip` was on `PATH`. It now runs a lightweight version/help probe
    and fails the H100 env smoke if the compute-built `lrzip` binary cannot
    execute on the allocated H100 node.
26. Static parser audit found that `parse_train_log.py` previously reported the
    last periodic train-loss step rather than the actual wallclock stop step.
    It now parses `stopping_early: wallclock_cap ... step: N/M` and reports
    `train_steps_final`. On the known 2026-04-27 seed-42 log it now reports
    `4945`, matching the record README.
27. Static runner audit found that candidate workloads were launched with direct
    `torchrun` inside the batch allocation. `run_candidate.sh` now wraps
    `torchrun` in `srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK` by
    default when `SLURM_JOB_ID` is present, preserving Slurm accounting while
    keeping `torchrun` responsible for the single-node 8-process launch.
28. Static timeout audit found that the one-hour record runner's inherited
    default candidate timeouts could add up to 65 minutes before env smoke or
    scratch staging. `h100-record-runner.sbatch` now defaults to 8-minute smoke
    timeouts and a 36-minute full-candidate timeout, for a 52-minute
    candidate-timeout ceiling. `h100-short-smoke.sbatch` now defaults to
    10-minute smoke timeouts inside its 45-minute allocation.
29. Candidate `status.json` now records `timeout` and `timed_out` in addition
    to `exit_code` and artifact-size fields, making timeout kills explicit in
    the H100 run summaries.
30. Static final-status audit found early env/staging failure paths that could
    exit before writing `final-status.json`. `goal-3/scripts/common.sh` now
    provides `goal3_write_final_status`, and the H100 env, short-smoke, and
    record-runner scripts install traps that write failure status when the
    normal summary path has not already done so.
31. Follow-up final-status audit moved the short-smoke and record-runner traps
    ahead of env-smoke execution and added the same bounded final-status trap to
    the optional repair-agent. Normal short-smoke and record-runner summaries
    now include candidate order, seed, timeout settings, and per-candidate
    status payloads.
32. The local `hpc` alias is not available directly in the Mac shell. The
    successful noninteractive sync/static-check path uses SSH ProxyJump through
    `peterj29@access.engr.oregonstate.edu` to
    `peterj29@submit-a.hpc.engr.oregonstate.edu`.
33. Added `goal-3/scripts/static_goal3_audit.py`, a text-level guardrail check
    that avoids importing H100-only dependencies while verifying qMLP
    flag/component wiring, Hessian/compression keys, candidate mappings, H100
    80GB constraints, final-status traps, scratch staging, and bounded runner
    defaults. It passed locally and on the remote submit node after removing a
    Python future import unsupported by the submit-node `python3`.

## Not Yet Known

- Whether qMLP stays beneficial inside the full 2026-04-27 record stack.
- Whether qMLP interacts cleanly with TTT LoRA, GPTQ/LQER, and per-group
  compression at runtime.
- Whether `sp16384` remains under 16 MB in the full record stack after qMLP,
  LQER, TTT hooks, and code-size changes.
- Whether OSU's H100/FA3/lrzip environment can reproduce the record stack
  without dependency or kernel issues.
- Whether qMLP matrix materialization adds enough overhead to reduce H100 step
  count materially versus the dense record.
- Whether the `sp16384` tokenizer loads with vocab size 16384 in the eventual
  H100/Goal 3 Python environment.
- Whether scratch staging overhead is small enough relative to the one-hour
  allocation. The datasets are small enough to stage in principle, but the real
  copy time should be visible in `scratch-stage.txt` and job logs.

## Current Conclusion

The next useful work is to ask for explicit approval to submit only the
15-minute H100 env smoke described in `goal-3/7-approval.md`. The CPU
environment and tools prep are complete, live Slurm state has been refreshed,
the exact H100 env-smoke request has a current `srun --test-only` estimate, and
the synced remote Goal 3 scripts pass submit-node static checks including the
static Goal 3 guardrail audit. No H100/H200 submission should happen until the
user approves that exact smoke request.
