Please execute the full quaternion Parameter Golf plan using the loop defined in `goal/0-loop.md` and the phase roadmap in `goal/0-plan.md`.

For each phase in `goal/0-plan.md`:

1. First create a detailed phase plan in `goal/` following the format and naming rules from `goal/0-loop.md`: `[PHASE-INDEX]-[ONE-WORD-DESCRIPTOR].md`.
2. Then implement that phase.
3. Then update `goal/0-plan.md`, `goal/0-loop.md`, the current phase file, and any earlier phase files whose assumptions changed, so the docs match the facts on the ground.
4. Then decide whether to continue, revise, repeat narrowly, block, abandon, or move to the next phase.

Use `goal/0-loop.md` as the operating procedure and `goal/0-plan.md` as the starting roadmap, but do not treat either file as infallible. Follow current evidence, live cluster state, code behavior, benchmark results, and your engineering judgment over stale assumptions in the plan. If the facts on the ground contradict the plan, update the documents and choose the path that best answers the core question: whether quaternion MLPs help lower Parameter Golf validation bits-per-byte after reinvesting saved parameters.

Watch carefully for impossible goals, unverifiable completion requirements, over-strict verification, hardware assumptions that no longer hold, and token-wasting loops. When a requirement is impossible or no longer useful, revise it into the smallest factual test that moves the project forward.

Keep the work focused on learning whether qMLP is a good tool for Parameter Golf. Prefer fast, cheap, well-recorded experiments before scarce or expensive runs. Use OSU/HPC resources safely according to the plan: no training or GPU-heavy work on submit nodes, use Slurm for compute work, and ask for approval before large/long/multi-GPU jobs or destructive actions.

Proceed phase by phase until the plan is complete, blocked by a real external condition, or the evidence is strong enough to make a defensible keep/modify/abandon decision on the quaternion MLP path.
