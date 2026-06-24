#!/usr/bin/env python3
"""Static guardrail checks for the Goal 3 H100 qMLP campaign.

This intentionally avoids importing the record stack, because that import can
touch CUDA/H100-specific dependencies. It checks the text invariants that would
otherwise be easy to break while editing the staged record script or runner.
"""

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "stage" / "primary-qmlp" / "train_gpt.py"
COMMON = ROOT / "scripts" / "common.sh"
ENV_SMOKE = ROOT / "h100-env-smoke.sbatch"
SHORT_SMOKE = ROOT / "h100-short-smoke.sbatch"
RECORD_RUNNER = ROOT / "h100-record-runner.sbatch"
CAMPAIGN_RUNNER = ROOT / "h100-campaign-runner.sbatch"
REPAIR_AGENT = ROOT / "h100-repair-agent.sbatch"
LOGS_KEEP = ROOT / "logs" / ".gitkeep"
RUN_CANDIDATE = ROOT / "scripts" / "run_candidate.sh"


def read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing required file: {path.relative_to(ROOT)}")
    return path.read_text()


def require(text: str, pattern: str, label: str) -> None:
    if not re.search(pattern, text, re.MULTILINE | re.DOTALL):
        raise AssertionError(f"missing invariant: {label}")


def require_literal(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing invariant: {label}")


def check_train(train: str) -> None:
    require_literal(train, 'quat_mlp = bool(int(os.environ.get("QUAT_MLP", "0")))', "QUAT_MLP flag")
    require_literal(train, 'quat_mlp_impl = os.environ.get("QUAT_MLP_IMPL", "matrix").lower()', "matrix qMLP default")
    require_literal(train, '_QUAT_COMPONENTS = ("wr", "wi", "wj", "wk")', "qMLP component names")
    require_literal(train, "def _quaternion_matrix(w):", "qMLP matrix materialization")
    require_literal(train, "def _quaternion_input_hessian(x):", "qMLP Hessian reduction")
    require_literal(train, "if self.quat_mlp and h.quat_mlp_impl != \"matrix\":", "unsupported qMLP implementation guard")
    require_literal(train, "torch.empty(h.num_layers, 4, hidden_dim // 4, h.model_dim // 4)", "qMLP up bank shape")
    require_literal(train, "torch.empty(h.num_layers, 4, h.model_dim // 4, hidden_dim // 4)", "qMLP down bank shape")
    require_literal(train, "if up_w.ndim == 3:", "MLP qMLP/dense dispatch")
    require_literal(train, "force_quantize = _is_quat_mlp_component(name)", "qMLP force quantization")
    require_literal(train, "quat_mlp=h.quat_mlp", "qMLP rebank during deserialize")

    for suffix in ("wr", "wi", "wj", "wk"):
        require_literal(train, f'blocks.{{layer_idx}}.mlp.fc.{{suffix}}', f"qMLP fc Hessian key {suffix}")
        require_literal(train, f'blocks.{{layer_idx}}.mlp.proj.{{suffix}}', f"qMLP proj Hessian key {suffix}")
        require_literal(train, f"mlp.fc.{suffix}.q", f"qMLP fc pergroup key {suffix}")
        require_literal(train, f"mlp.proj.{suffix}.q", f"qMLP proj pergroup key {suffix}")
        require_literal(train, f'blocks.{{i}}.mlp.fc.{{suffix}}', f"qMLP unbank fc key {suffix}")
        require_literal(train, f'blocks.{{i}}.mlp.proj.{{suffix}}', f"qMLP unbank proj key {suffix}")


def check_common(common: str) -> None:
    require_literal(common, "GOAL3_SP8192_DATA=", "sp8192 data path")
    require_literal(common, "GOAL3_SP16384_DATA=", "sp16384 data path")
    require_literal(common, "goal3_write_final_status()", "final-status helper")
    require_literal(common, "goal3_prepare_runtime_storage()", "runtime scratch/cache helper")
    require_literal(common, "goal3_prepare_local_workspace()", "scratch staging helper")
    require_literal(common, "goal3_ensure_runtime_requirements()", "runtime requirements helper")
    require_literal(common, "goal3_record_source_snapshot()", "source snapshot helper")
    require_literal(common, "TORCHINDUCTOR_CACHE_DIR", "torch compile cache routed to scratch")
    require_literal(common, "TRITON_CACHE_DIR", "triton cache routed to scratch")
    require_literal(common, "PIP_CACHE_DIR", "pip cache routed to scratch")
    require_literal(common, "TMPDIR", "runtime temp routed to scratch")

    require(
        common,
        r"dense_sp8192\|dense_sp8192_smoke\).*?export VOCAB_SIZE=8192.*?export QUAT_MLP=0",
        "dense sp8192 candidate maps to QUAT_MLP=0",
    )
    require(
        common,
        r"qmlp_sp8192\|qmlp_sp8192_smoke\).*?export VOCAB_SIZE=8192.*?export QUAT_MLP=1",
        "qMLP sp8192 candidate maps to QUAT_MLP=1",
    )
    require(
        common,
        r"qmlp_sp16384\|qmlp_sp16384_smoke\|qmlp_sp16384_ttt_smoke\).*?export VOCAB_SIZE=16384.*?export QUAT_MLP=1",
        "qMLP sp16384 candidate maps to QUAT_MLP=1",
    )
    require_literal(common, 'if [[ "$candidate" == *_ttt_smoke ]]; then', "TTT smoke candidate branch")
    require_literal(common, "export TTT_ENABLED=1", "TTT smoke re-enables TTT")
    require_literal(common, "export TTT_EVAL_BATCHES=${TTT_EVAL_BATCHES:-1}", "TTT smoke bounds eval batches")


def check_sbatch(name: str, text: str, *, expect_candidate_order: bool = False) -> None:
    require_literal(text, "#SBATCH --partition=dgxh", f"{name} uses dgxh")
    require_literal(text, "#SBATCH --constraint=h100&vram80g", f"{name} uses H100 80GB constraint")
    require_literal(text, "#SBATCH --gres=gpu:8", f"{name} requests 8 GPUs")
    require_literal(text, "goal3_init_run_dir", f"{name} initializes run dir")
    require_literal(text, "goal3_prepare_runtime_storage", f"{name} prepares scratch runtime cache")
    require_literal(text, "goal3_record_source_snapshot", f"{name} records source snapshot")
    require_literal(text, "trap cleanup EXIT TERM INT", f"{name} has final-status trap")
    if name == "campaign-runner":
        require_literal(text, "write_campaign_final_status", f"{name} writes rich final status")
    else:
        require_literal(text, "goal3_write_final_status", f"{name} writes final status")
    if name != "repair-agent":
        require_literal(text, "goal3_activate_env", f"{name} activates env")
        require_literal(text, "env_smoke.py", f"{name} runs env smoke")
    if name in {"short-smoke", "record-runner"}:
        require_literal(text, "goal3_prepare_local_workspace", f"{name} stages scratch workspace")
        require_literal(text, "run_candidate.sh", f"{name} delegates candidates")
    if name == "short-smoke":
        require_literal(
            text,
            'smoke_candidates=${GOAL3_SMOKE_CANDIDATES:-"dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384_smoke qmlp_sp16384_ttt_smoke"}',
            "short-smoke defaults to dense/qMLP/qMLP+TTT smoke gates",
        )
    if expect_candidate_order:
        require_literal(
            text,
            'dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384',
            f"{name} default candidate order",
        )
        require_literal(text, "GOAL3_SMOKE_TIMEOUT:-8m", f"{name} bounded smoke timeout")
        require_literal(text, "GOAL3_FULL_TIMEOUT:-36m", f"{name} bounded full timeout")
    if name == "campaign-runner":
        require_literal(text, "#SBATCH --time=06:00:00", "campaign requests padded queue slot")
        require_literal(text, "write_campaign_final_status()", "campaign has rich final-status writer")
        require_literal(text, "goal3_ensure_runtime_requirements", "campaign validates/builds runtime requirements")
        require_literal(
            text,
            'smoke_candidates=${GOAL3_SMOKE_CANDIDATES:-"dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384_smoke qmlp_sp16384_ttt_smoke"}',
            "campaign defaults to dense/qMLP smoke gates",
        )
        require_literal(text, "smoke-gate.json", "campaign records smoke-gate result")
        require_literal(text, "_load_error", "campaign gate reports malformed json")
        require_literal(text, "missing summary.json", "campaign baseline gate reports missing summary")
        require_literal(text, "attempted_smoke_candidates", "campaign tracks attempted smoke candidates")
        require_literal(text, "not_attempted", "campaign records unattempted smokes after failure")
        require_literal(text, "stopping smoke gate after first failed smoke", "campaign stops smoke gate on first failure")
        require_literal(text, "GOAL3_SMOKE_TIMEOUT:-20m", "campaign has padded smoke timeout")
        require_literal(text, "baseline-parity.json", "campaign writes baseline parity record")
        require_literal(text, "baseline_status=$?", "campaign captures baseline exit status before parity gate")
        require_literal(text, "baseline candidate $baseline_candidate seed $baseline_seed failed", "campaign logs failed baseline candidate")
        require_literal(text, "GOAL3_BASELINE_PARITY_MAX_BPB", "campaign has baseline parity BPB target")
        require_literal(text, "GOAL3_BASELINE_HARD_MAX_BPB", "campaign has hard baseline BPB gate")
        require_literal(text, "GOAL3_BASELINE_HARD_MIN_STEPS", "campaign has hard baseline step gate")
        require_literal(text, "strict_parity_passed", "campaign records strict parity separately")
        require_literal(text, "GOAL3_FULL_TIMEOUT:-120m", "campaign has padded full-candidate timeout")
        require_literal(text, 'qmlp_seeds=${GOAL3_QMLP_SEEDS:-"42 0 1234"}', "campaign defaults to three qMLP seeds")
        require_literal(text, "qmlp_quantized_ttt_val_bpb_mean", "campaign summarizes qMLP mean")
        require_literal(text, "source_snapshot_manifest", "campaign final status points to source snapshot manifest")


def check_run_candidate(text: str) -> None:
    require_literal(
        text,
        'export RUN_ID=${GOAL3_RUN_ID_OVERRIDE:-${candidate}_seed${seed}}',
        "candidate run id is candidate/seed scoped by default",
    )
    require_literal(text, "export ARTIFACT_DIR=$candidate_dir", "candidate artifact dir scoped by candidate/seed")


def main() -> int:
    try:
        train = read(TRAIN)
        common = read(COMMON)
        check_train(train)
        check_common(common)
        check_run_candidate(read(RUN_CANDIDATE))
        check_sbatch("env-smoke", read(ENV_SMOKE))
        check_sbatch("short-smoke", read(SHORT_SMOKE))
        check_sbatch("record-runner", read(RECORD_RUNNER), expect_candidate_order=True)
        check_sbatch("campaign-runner", read(CAMPAIGN_RUNNER))
        check_sbatch("repair-agent", read(REPAIR_AGENT))
        if not LOGS_KEEP.exists():
            raise AssertionError("missing invariant: logs/.gitkeep for sbatch stdout/stderr directory")
    except AssertionError as exc:
        print(f"static_goal3_audit: FAILED: {exc}", file=sys.stderr)
        return 1
    print("static_goal3_audit: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
