#!/usr/bin/env bash
#SBATCH --job-name=goal3-h100-campaign
#SBATCH --partition=dgxh
#SBATCH --constraint=h100&vram80g
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --output=goal-3/logs/%x-%j.out
#SBATCH --error=goal-3/logs/%x-%j.err

set -euo pipefail
umask 077

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm
fi

repo_root=${GOAL3_REPO_ROOT:-/nfs/hpc/share/peterj29/pg/src/pg}
script_dir="$repo_root/goal-3/scripts"
# shellcheck disable=SC1091
source "$script_dir/common.sh"

goal3_init_run_dir

write_campaign_final_status() {
    local status=${1:-1}
    local reason=${2:-unknown}
    python - "$GOAL3_RUN_DIR" "$status" "$reason" "$(goal3_timestamp)" <<'PY'
import json
import statistics
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
exit_code = int(sys.argv[2])
reason = sys.argv[3]
timestamp = sys.argv[4]
final_path = run_dir / "final-status.json"
if final_path.exists():
    raise SystemExit(0)


def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_load_error": repr(exc), "_path": str(path)}


def read_text(path):
    if not path.exists():
        return None
    return path.read_text(errors="replace")


runs = []
candidates_dir = run_dir / "candidates"
if candidates_dir.exists():
    for seed_dir in sorted(candidates_dir.glob("*/seed_*")):
        runs.append({
            "candidate": seed_dir.parent.name,
            "seed": seed_dir.name.removeprefix("seed_"),
            "env": read_text(seed_dir / "env.txt"),
            "summary": load_json(seed_dir / "summary.json"),
            "status": load_json(seed_dir / "status.json"),
            "artifacts": load_json(seed_dir / "artifacts.json"),
        })

qmlp_bpbs = [
    run["summary"].get("quantized_ttt_val_bpb")
    for run in runs
    if (
        run["candidate"].startswith("qmlp_")
        and not run["candidate"].endswith("_smoke")
        and isinstance(run["summary"], dict)
    )
]
qmlp_bpbs = [x for x in qmlp_bpbs if x is not None]

payload = {
    "status": "passed" if exit_code == 0 else "failed",
    "exit_code": exit_code,
    "reason": reason,
    "timestamp": timestamp,
    "campaign": "baseline_parity_then_qmlp_3seed",
    "env_smoke": load_json(run_dir / "env-smoke.json"),
    "smoke_gate": load_json(run_dir / "smoke-gate.json"),
    "baseline_parity": load_json(run_dir / "baseline-parity.json"),
    "runs": runs,
    "qmlp_quantized_ttt_val_bpb_mean": statistics.mean(qmlp_bpbs) if qmlp_bpbs else None,
    "qmlp_quantized_ttt_val_bpb_stdev": statistics.stdev(qmlp_bpbs) if len(qmlp_bpbs) > 1 else None,
    "files": {
        "context": str(run_dir / "context.txt") if (run_dir / "context.txt").exists() else None,
        "gpu": str(run_dir / "gpu.txt") if (run_dir / "gpu.txt").exists() else None,
        "python": str(run_dir / "python.txt") if (run_dir / "python.txt").exists() else None,
        "run_order": str(run_dir / "run-order.txt") if (run_dir / "run-order.txt").exists() else None,
        "progress": str(run_dir / "progress.txt") if (run_dir / "progress.txt").exists() else None,
        "scratch_stage": str(run_dir / "scratch-stage.txt") if (run_dir / "scratch-stage.txt").exists() else None,
        "source_snapshot_manifest": str(run_dir / "source-snapshot.sha256")
        if (run_dir / "source-snapshot.sha256").exists()
        else None,
    },
}
final_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

cleanup() {
    status=$?
    trap - EXIT TERM INT
    write_campaign_final_status "$status" "$([[ "$status" -eq 0 ]] && echo passed || echo early_exit)" || true
    goal3_cleanup_local_workspace || true
    exit "$status"
}
trap cleanup EXIT TERM INT

goal3_prepare_runtime_storage
goal3_activate_env
goal3_ensure_runtime_requirements "$GOAL3_RUN_DIR"
goal3_record_context "$GOAL3_RUN_DIR"
goal3_record_source_snapshot "$GOAL3_RUN_DIR"
goal3_record_gpu_context "$GOAL3_RUN_DIR"
goal3_record_python_context "$GOAL3_RUN_DIR"

export GOAL3_EXPECT_CUDA_DEVICES=8
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python "$script_dir/env_smoke.py" "$GOAL3_RUN_DIR/env-smoke.json"

goal3_prepare_local_workspace

export GOAL3_FULL_TIMEOUT=${GOAL3_FULL_TIMEOUT:-120m}
export GOAL3_SMOKE_TIMEOUT=${GOAL3_SMOKE_TIMEOUT:-20m}
export GOAL3_BASELINE_PARITY_MAX_BPB=${GOAL3_BASELINE_PARITY_MAX_BPB:-1.065}
export GOAL3_BASELINE_HARD_MAX_BPB=${GOAL3_BASELINE_HARD_MAX_BPB:-1.075}
export GOAL3_BASELINE_PARITY_MIN_STEPS=${GOAL3_BASELINE_PARITY_MIN_STEPS:-4500}
export GOAL3_BASELINE_HARD_MIN_STEPS=${GOAL3_BASELINE_HARD_MIN_STEPS:-4000}

baseline_candidate=${GOAL3_BASELINE_CANDIDATE:-dense_sp8192}
baseline_seed=${GOAL3_BASELINE_SEED:-42}
smoke_seed=${GOAL3_SMOKE_SEED:-42}
smoke_candidates=${GOAL3_SMOKE_CANDIDATES:-"dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384_smoke qmlp_sp16384_ttt_smoke"}
qmlp_candidate=${GOAL3_QMLP_CANDIDATE:-qmlp_sp16384}
qmlp_seeds=${GOAL3_QMLP_SEEDS:-"42 0 1234"}

{
    echo "campaign=baseline_parity_then_qmlp_3seed"
    echo "smoke_candidates=$smoke_candidates"
    echo "smoke_seed=$smoke_seed"
    echo "baseline_candidate=$baseline_candidate"
    echo "baseline_seed=$baseline_seed"
    echo "qmlp_candidate=$qmlp_candidate"
    echo "qmlp_seeds=$qmlp_seeds"
    echo "full_timeout=$GOAL3_FULL_TIMEOUT"
    echo "smoke_timeout=$GOAL3_SMOKE_TIMEOUT"
    echo "baseline_parity_max_bpb=$GOAL3_BASELINE_PARITY_MAX_BPB"
    echo "baseline_hard_max_bpb=$GOAL3_BASELINE_HARD_MAX_BPB"
    echo "baseline_parity_min_steps=$GOAL3_BASELINE_PARITY_MIN_STEPS"
    echo "baseline_hard_min_steps=$GOAL3_BASELINE_HARD_MIN_STEPS"
} >"$GOAL3_RUN_DIR/run-order.txt"

attempted_smoke_candidates=""
for candidate in $smoke_candidates; do
    attempted_smoke_candidates="${attempted_smoke_candidates:+$attempted_smoke_candidates }$candidate"
    echo "starting smoke $candidate seed $smoke_seed at $(goal3_timestamp)" \
        | tee -a "$GOAL3_RUN_DIR/progress.txt"
    set +e
    bash "$script_dir/run_candidate.sh" "$candidate" "$smoke_seed"
    smoke_status=$?
    set -e
    if [[ "$smoke_status" -ne 0 ]]; then
        echo "smoke $candidate seed $smoke_seed failed with exit $smoke_status at $(goal3_timestamp)" \
            | tee -a "$GOAL3_RUN_DIR/progress.txt"
        echo "stopping smoke gate after first failed smoke at $(goal3_timestamp)" \
            | tee -a "$GOAL3_RUN_DIR/progress.txt"
        break
    fi
    echo "finished smoke $candidate seed $smoke_seed at $(goal3_timestamp)" \
        | tee -a "$GOAL3_RUN_DIR/progress.txt"
done

python - "$GOAL3_RUN_DIR" "$smoke_candidates" "$attempted_smoke_candidates" \
    "$smoke_seed" "$GOAL3_SMOKE_TIMEOUT" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
candidates = sys.argv[2].split()
attempted = set(sys.argv[3].split())
seed = sys.argv[4]
smoke_timeout = sys.argv[5]


def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_load_error": repr(exc), "_path": str(path)}


statuses = []
reasons = []
for candidate in candidates:
    base = run_dir / "candidates" / candidate / f"seed_{seed}"
    status_path = base / "status.json"
    summary_path = base / "summary.json"
    if candidate not in attempted:
        statuses.append({
            "candidate": candidate,
            "seed": seed,
            "status": {"not_attempted": True},
            "summary": None,
        })
        continue
    status = load_json(status_path)
    summary = load_json(summary_path)
    statuses.append({
        "candidate": candidate,
        "seed": seed,
        "status": status,
        "summary": summary,
    })
    if status is None:
        reasons.append(f"{candidate}: missing status.json")
    elif "_load_error" in status:
        reasons.append(f"{candidate}: invalid status.json")
    elif status.get("exit_code") != 0:
        reasons.append(f"{candidate}: exit_code={status.get('exit_code')}")
    elif status.get("total_submission_bytes") is not None and status.get("artifact_under_limit") is not True:
        reasons.append(f"{candidate}: artifact not under limit")
payload = {
    "candidate_order": candidates,
    "attempted_candidates": [candidate for candidate in candidates if candidate in attempted],
    "seed": seed,
    "smoke_timeout": smoke_timeout,
    "candidate_statuses": statuses,
    "passed": not reasons,
    "reasons": reasons,
}
(run_dir / "smoke-gate.json").write_text(json.dumps(payload, indent=2) + "\n")
if reasons:
    print(json.dumps(payload, indent=2))
    raise SystemExit(60)
PY

echo "starting baseline parity $baseline_candidate seed $baseline_seed at $(goal3_timestamp)" \
    | tee -a "$GOAL3_RUN_DIR/progress.txt"
set +e
bash "$script_dir/run_candidate.sh" "$baseline_candidate" "$baseline_seed"
baseline_status=$?
set -e
if [[ "$baseline_status" -ne 0 ]]; then
    echo "baseline candidate $baseline_candidate seed $baseline_seed failed with exit $baseline_status at $(goal3_timestamp)" \
        | tee -a "$GOAL3_RUN_DIR/progress.txt"
fi
echo "finished baseline parity $baseline_candidate seed $baseline_seed at $(goal3_timestamp)" \
    | tee -a "$GOAL3_RUN_DIR/progress.txt"

python - "$GOAL3_RUN_DIR" "$baseline_candidate" "$baseline_seed" \
    "$GOAL3_BASELINE_PARITY_MAX_BPB" "$GOAL3_BASELINE_HARD_MAX_BPB" \
    "$GOAL3_BASELINE_PARITY_MIN_STEPS" "$GOAL3_BASELINE_HARD_MIN_STEPS" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
candidate = sys.argv[2]
seed = sys.argv[3]
target_max_bpb = float(sys.argv[4])
hard_max_bpb = float(sys.argv[5])
target_min_steps = int(sys.argv[6])
hard_min_steps = int(sys.argv[7])
base = run_dir / "candidates" / candidate / f"seed_{seed}"


def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_load_error": repr(exc), "_path": str(path)}


summary = load_json(base / "summary.json")
status = load_json(base / "status.json")
hard_reasons = []
warnings = []
if status is None:
    hard_reasons.append("missing status.json")
    status = {}
elif "_load_error" in status:
    hard_reasons.append("invalid status.json")
if summary is None:
    hard_reasons.append("missing summary.json")
    summary = {}
elif "_load_error" in summary:
    hard_reasons.append("invalid summary.json")
if status.get("exit_code") != 0:
    hard_reasons.append(f"exit_code={status.get('exit_code')}")
if status.get("artifact_under_limit") is not True:
    hard_reasons.append("artifact not proven under limit")
bpb = summary.get("quantized_ttt_val_bpb")
if bpb is None:
    hard_reasons.append("quantized_ttt_val_bpb missing")
elif bpb > hard_max_bpb:
    hard_reasons.append(f"quantized_ttt_val_bpb={bpb} > hard_max_bpb={hard_max_bpb}")
elif bpb > target_max_bpb:
    warnings.append(f"quantized_ttt_val_bpb={bpb} > parity_target_bpb={target_max_bpb}")
steps = summary.get("train_steps_final")
if steps is None:
    hard_reasons.append("train_steps_final missing")
elif steps < hard_min_steps:
    hard_reasons.append(f"train_steps_final={steps} < hard_min_steps={hard_min_steps}")
elif steps < target_min_steps:
    warnings.append(f"train_steps_final={steps} < parity_target_min_steps={target_min_steps}")
payload = {
    "candidate": candidate,
    "seed": seed,
    "parity_target_max_bpb": target_max_bpb,
    "hard_max_bpb": hard_max_bpb,
    "parity_target_min_steps": target_min_steps,
    "hard_min_steps": hard_min_steps,
    "summary": summary,
    "status": status,
    "passed": not hard_reasons,
    "strict_parity_passed": not hard_reasons and not warnings,
    "hard_reasons": hard_reasons,
    "warnings": warnings,
}
(run_dir / "baseline-parity.json").write_text(json.dumps(payload, indent=2) + "\n")
if hard_reasons:
    print(json.dumps(payload, indent=2))
    raise SystemExit(70)
PY

for seed in $qmlp_seeds; do
    echo "starting qMLP candidate $qmlp_candidate seed $seed at $(goal3_timestamp)" \
        | tee -a "$GOAL3_RUN_DIR/progress.txt"
    bash "$script_dir/run_candidate.sh" "$qmlp_candidate" "$seed"
    echo "finished qMLP candidate $qmlp_candidate seed $seed at $(goal3_timestamp)" \
        | tee -a "$GOAL3_RUN_DIR/progress.txt"
done

python - "$GOAL3_RUN_DIR" "$baseline_candidate" "$baseline_seed" "$qmlp_candidate" "$qmlp_seeds" <<'PY'
import json
import statistics
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
baseline_candidate = sys.argv[2]
baseline_seed = sys.argv[3]
qmlp_candidate = sys.argv[4]
qmlp_seeds = sys.argv[5].split()


def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_load_error": repr(exc), "_path": str(path)}


def load_candidate(candidate, seed):
    base = run_dir / "candidates" / candidate / f"seed_{seed}"
    summary_path = base / "summary.json"
    status_path = base / "status.json"
    artifacts_path = base / "artifacts.json"

    return {
        "candidate": candidate,
        "seed": seed,
        "summary": load_json(summary_path),
        "status": load_json(status_path),
        "artifacts": load_json(artifacts_path),
    }

runs = [load_candidate(baseline_candidate, baseline_seed)]
runs.extend(load_candidate(qmlp_candidate, seed) for seed in qmlp_seeds)
qmlp_bpbs = [
    run["summary"].get("quantized_ttt_val_bpb")
    for run in runs
    if run["candidate"] == qmlp_candidate and isinstance(run["summary"], dict)
]
qmlp_bpbs = [x for x in qmlp_bpbs if x is not None]
final = {
    "status": "passed",
    "exit_code": 0,
    "campaign": "baseline_parity_then_qmlp_3seed",
    "env_smoke": load_json(run_dir / "env-smoke.json"),
    "smoke_gate": load_json(run_dir / "smoke-gate.json"),
    "baseline_parity": load_json(run_dir / "baseline-parity.json"),
    "runs": runs,
    "qmlp_quantized_ttt_val_bpb_mean": statistics.mean(qmlp_bpbs) if qmlp_bpbs else None,
    "qmlp_quantized_ttt_val_bpb_stdev": statistics.stdev(qmlp_bpbs) if len(qmlp_bpbs) > 1 else None,
    "files": {
        "context": str(run_dir / "context.txt") if (run_dir / "context.txt").exists() else None,
        "gpu": str(run_dir / "gpu.txt") if (run_dir / "gpu.txt").exists() else None,
        "python": str(run_dir / "python.txt") if (run_dir / "python.txt").exists() else None,
        "run_order": str(run_dir / "run-order.txt") if (run_dir / "run-order.txt").exists() else None,
        "progress": str(run_dir / "progress.txt") if (run_dir / "progress.txt").exists() else None,
        "scratch_stage": str(run_dir / "scratch-stage.txt") if (run_dir / "scratch-stage.txt").exists() else None,
        "source_snapshot_manifest": str(run_dir / "source-snapshot.sha256")
        if (run_dir / "source-snapshot.sha256").exists()
        else None,
    },
}
(run_dir / "final-status.json").write_text(json.dumps(final, indent=2) + "\n")
PY
