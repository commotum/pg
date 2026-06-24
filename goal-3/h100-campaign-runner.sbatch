#!/usr/bin/env bash
#SBATCH --job-name=goal3-h100-campaign
#SBATCH --partition=dgxh
#SBATCH --constraint=h100&vram80g
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
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

cleanup() {
    status=$?
    trap - EXIT TERM INT
    if [[ "$status" -eq 0 ]]; then
        goal3_write_final_status 0 passed || true
    else
        goal3_write_final_status "$status" early_exit || true
    fi
    goal3_cleanup_local_workspace || true
    exit "$status"
}
trap cleanup EXIT TERM INT

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

export GOAL3_FULL_TIMEOUT=${GOAL3_FULL_TIMEOUT:-30m}
export GOAL3_BASELINE_PARITY_MAX_BPB=${GOAL3_BASELINE_PARITY_MAX_BPB:-1.065}
export GOAL3_BASELINE_PARITY_MIN_STEPS=${GOAL3_BASELINE_PARITY_MIN_STEPS:-4500}

baseline_candidate=${GOAL3_BASELINE_CANDIDATE:-dense_sp8192}
baseline_seed=${GOAL3_BASELINE_SEED:-42}
qmlp_candidate=${GOAL3_QMLP_CANDIDATE:-qmlp_sp16384}
qmlp_seeds=${GOAL3_QMLP_SEEDS:-"42 0 1234"}

{
    echo "campaign=baseline_parity_then_qmlp_3seed"
    echo "baseline_candidate=$baseline_candidate"
    echo "baseline_seed=$baseline_seed"
    echo "qmlp_candidate=$qmlp_candidate"
    echo "qmlp_seeds=$qmlp_seeds"
    echo "full_timeout=$GOAL3_FULL_TIMEOUT"
    echo "baseline_parity_max_bpb=$GOAL3_BASELINE_PARITY_MAX_BPB"
    echo "baseline_parity_min_steps=$GOAL3_BASELINE_PARITY_MIN_STEPS"
} >"$GOAL3_RUN_DIR/run-order.txt"

echo "starting baseline parity $baseline_candidate seed $baseline_seed at $(goal3_timestamp)" \
    | tee -a "$GOAL3_RUN_DIR/progress.txt"
bash "$script_dir/run_candidate.sh" "$baseline_candidate" "$baseline_seed"
echo "finished baseline parity $baseline_candidate seed $baseline_seed at $(goal3_timestamp)" \
    | tee -a "$GOAL3_RUN_DIR/progress.txt"

python - "$GOAL3_RUN_DIR" "$baseline_candidate" "$baseline_seed" \
    "$GOAL3_BASELINE_PARITY_MAX_BPB" "$GOAL3_BASELINE_PARITY_MIN_STEPS" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
candidate = sys.argv[2]
seed = sys.argv[3]
max_bpb = float(sys.argv[4])
min_steps = int(sys.argv[5])
base = run_dir / "candidates" / candidate / f"seed_{seed}"
summary = json.loads((base / "summary.json").read_text())
status = json.loads((base / "status.json").read_text())
reasons = []
if status.get("exit_code") != 0:
    reasons.append(f"exit_code={status.get('exit_code')}")
if status.get("artifact_under_limit") is not True:
    reasons.append("artifact not proven under limit")
bpb = summary.get("quantized_ttt_val_bpb")
if bpb is None or bpb > max_bpb:
    reasons.append(f"quantized_ttt_val_bpb={bpb} > {max_bpb}")
steps = summary.get("train_steps_final")
if steps is None or steps < min_steps:
    reasons.append(f"train_steps_final={steps} < {min_steps}")
payload = {
    "candidate": candidate,
    "seed": seed,
    "max_bpb": max_bpb,
    "min_steps": min_steps,
    "summary": summary,
    "status": status,
    "passed": not reasons,
    "reasons": reasons,
}
(run_dir / "baseline-parity.json").write_text(json.dumps(payload, indent=2) + "\n")
if reasons:
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

def load_candidate(candidate, seed):
    base = run_dir / "candidates" / candidate / f"seed_{seed}"
    summary_path = base / "summary.json"
    status_path = base / "status.json"
    artifacts_path = base / "artifacts.json"
    return {
        "candidate": candidate,
        "seed": seed,
        "summary": json.loads(summary_path.read_text()) if summary_path.exists() else None,
        "status": json.loads(status_path.read_text()) if status_path.exists() else None,
        "artifacts": json.loads(artifacts_path.read_text()) if artifacts_path.exists() else None,
    }

runs = [load_candidate(baseline_candidate, baseline_seed)]
runs.extend(load_candidate(qmlp_candidate, seed) for seed in qmlp_seeds)
qmlp_bpbs = [
    run["summary"].get("quantized_ttt_val_bpb")
    for run in runs
    if run["candidate"] == qmlp_candidate and run["summary"] is not None
]
qmlp_bpbs = [x for x in qmlp_bpbs if x is not None]
final = {
    "status": "passed",
    "exit_code": 0,
    "campaign": "baseline_parity_then_qmlp_3seed",
    "baseline_parity": json.loads((run_dir / "baseline-parity.json").read_text()),
    "runs": runs,
    "qmlp_quantized_ttt_val_bpb_mean": statistics.mean(qmlp_bpbs) if qmlp_bpbs else None,
    "qmlp_quantized_ttt_val_bpb_stdev": statistics.stdev(qmlp_bpbs) if len(qmlp_bpbs) > 1 else None,
}
(run_dir / "final-status.json").write_text(json.dumps(final, indent=2) + "\n")
PY
