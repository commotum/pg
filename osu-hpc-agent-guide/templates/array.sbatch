#!/usr/bin/env bash
#SBATCH --job-name=osu-array-smoke
#SBATCH --partition=share
#SBATCH --array=0-9%2
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=osu-array-%A_%a.out
#SBATCH --error=osu-array-%A_%a.err

# `%2` limits this example to two concurrent elements.
# Submit as-is:
#   sbatch array.sbatch
#
# Or pass a command. The array index is available as SLURM_ARRAY_TASK_ID:
#   sbatch array.sbatch bash scripts/run-one.sh

set -euo pipefail
umask 077

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm
fi
# module load python/VERSION

submit_dir=${SLURM_SUBMIT_DIR:-$PWD}
if [[ -d /nfs/hpc/share/$USER ]]; then
    shared_root=${RUN_ROOT:-/nfs/hpc/share/$USER/agent-runs}
else
    shared_root=${RUN_ROOT:-$submit_dir/agent-runs}
fi
run_dir="$shared_root/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$run_dir"

{
    echo "array_job_id=$SLURM_ARRAY_JOB_ID"
    echo "array_task_id=$SLURM_ARRAY_TASK_ID"
    echo "job_id=$SLURM_JOB_ID"
    echo "host=$(hostname -f 2>/dev/null || hostname)"
    echo "git_commit=$(git -C "$submit_dir" rev-parse HEAD 2>/dev/null || true)"
    module -t list 2>&1 || true
} >"$run_dir/manifest.txt"

if [[ $# -gt 0 ]]; then
    srun "$@"
else
    srun bash -lc '
        set -euo pipefail
        printf "task=%s host=%s completed=%s\n" \
          "$SLURM_ARRAY_TASK_ID" \
          "$(hostname)" \
          "$(date --iso-8601=seconds 2>/dev/null || date)"
    ' >"$run_dir/result.txt"
fi
