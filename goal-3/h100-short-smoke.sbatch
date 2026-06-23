#!/usr/bin/env bash
#SBATCH --job-name=goal3-h100-smoke
#SBATCH --partition=dgxh
#SBATCH --constraint=h100&vram80g
#SBATCH --gres=gpu:8
#SBATCH --time=00:45:00
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
goal3_activate_env
goal3_record_context "$GOAL3_RUN_DIR"
goal3_record_gpu_context "$GOAL3_RUN_DIR"
goal3_record_python_context "$GOAL3_RUN_DIR"

export GOAL3_EXPECT_CUDA_DEVICES=8
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python "$script_dir/env_smoke.py" "$GOAL3_RUN_DIR/env-smoke.json"

cleanup() {
    status=$?
    trap - EXIT TERM INT
    goal3_cleanup_local_workspace || true
    exit "$status"
}
trap cleanup EXIT TERM INT

goal3_prepare_local_workspace

smoke_candidates=${GOAL3_SMOKE_CANDIDATES:-"dense_sp8192_smoke qmlp_sp8192_smoke qmlp_sp16384_smoke"}
printf 'smoke_candidates=%s\n' "$smoke_candidates" >"$GOAL3_RUN_DIR/run-order.txt"

for candidate in $smoke_candidates; do
    echo "starting $candidate at $(goal3_timestamp)" | tee -a "$GOAL3_RUN_DIR/progress.txt"
    bash "$script_dir/run_candidate.sh" "$candidate" "${GOAL3_SEED:-42}"
    echo "finished $candidate at $(goal3_timestamp)" | tee -a "$GOAL3_RUN_DIR/progress.txt"
done

printf '{"status":"passed","timestamp":"%s"}\n' "$(goal3_timestamp)" \
    >"$GOAL3_RUN_DIR/final-status.json"
