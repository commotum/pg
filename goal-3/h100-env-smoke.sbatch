#!/usr/bin/env bash
#SBATCH --job-name=goal3-h100-env
#SBATCH --partition=dgxh
#SBATCH --constraint=h100&vram80g
#SBATCH --gres=gpu:8
#SBATCH --time=00:15:00
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
    exit "$status"
}
trap cleanup EXIT TERM INT

goal3_activate_env
goal3_record_context "$GOAL3_RUN_DIR"
goal3_record_source_snapshot "$GOAL3_RUN_DIR"
goal3_record_gpu_context "$GOAL3_RUN_DIR"
goal3_record_python_context "$GOAL3_RUN_DIR"

export GOAL3_EXPECT_CUDA_DEVICES=8
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    python "$script_dir/env_smoke.py" "$GOAL3_RUN_DIR/env-smoke.json"

goal3_write_final_status 0 passed
