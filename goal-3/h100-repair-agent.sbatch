#!/usr/bin/env bash
#SBATCH --job-name=goal3-h100-repair
#SBATCH --partition=dgxh
#SBATCH --constraint=h100&vram80g
#SBATCH --gres=gpu:8
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --output=goal-3/logs/%x-%j.out
#SBATCH --error=goal-3/logs/%x-%j.err

set -euo pipefail
umask 077

if [[ "${GOAL3_ENABLE_REPAIR_AGENT:-0}" != "1" ]]; then
    echo "Repair agent is disabled. Set GOAL3_ENABLE_REPAIR_AGENT=1 only after review."
    exit 0
fi

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
        goal3_write_final_status 0 completed || true
    else
        goal3_write_final_status "$status" early_exit || true
    fi
    exit "$status"
}
trap cleanup EXIT TERM INT

goal3_activate_env || true
goal3_record_context "$GOAL3_RUN_DIR"
goal3_record_gpu_context "$GOAL3_RUN_DIR"

codex_bin=${GOAL3_CODEX_BIN:-/nfs/stak/users/peterj29/.local/bin/codex}
if [[ ! -x "$codex_bin" ]]; then
    echo "codex not executable at $codex_bin" >&2
    exit 2
fi

repair_prompt="$GOAL3_RUN_DIR/repair-prompt.md"
{
    cat "$repo_root/goal-3/0-prompt.md"
    printf '\n\n## Repair-Agent Bounds\n\n'
    printf '%s\n' '- You are running inside a bounded H100 repair job.'
    printf '%s\n' '- Do not submit new Slurm jobs.'
    printf '%s\n' '- Do not launch broad sweeps.'
    printf '%s\n' '- Only diagnose and patch the current Goal 3 run path.'
    printf '%s\n' '- Run at most one bounded smoke after a patch.'
    printf '%s\n' '- Write findings to goal-3/status.md and the current run directory.'
} >"$repair_prompt"

cd "$repo_root"
timeout "${GOAL3_REPAIR_TIMEOUT:-20m}" "$codex_bin" exec \
    --cd "$repo_root" \
    "$(cat "$repair_prompt")" \
    >"$GOAL3_RUN_DIR/codex-repair.stdout" \
    2>"$GOAL3_RUN_DIR/codex-repair.stderr"

goal3_write_final_status 0 completed
