#!/usr/bin/env bash
#SBATCH --job-name=osu-cpu-smoke
#SBATCH --partition=share
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=osu-cpu-%j.out
#SBATCH --error=osu-cpu-%j.err

# Submit as-is for a small smoke test:
#   sbatch cpu.sbatch
#
# Or pass a command:
#   sbatch cpu.sbatch python3 train.py --config config/small.yaml
#
# Add --account=NAME when a sponsored/private account is required.
# Pin project modules below after checking `module spider`.

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
run_dir="$shared_root/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
local_dir="/scratch/$USER/${SLURM_JOB_ID}"

mkdir -p "$run_dir/output" "$local_dir/output"

cleanup() {
    status=$?
    trap - EXIT TERM INT
    # Preserve any local outputs. A copy failure changes success to exit 90.
    if ! rsync -a --partial "$local_dir/output/" "$run_dir/output/"; then
        echo "stage-out failed" >&2
        [[ $status -ne 0 ]] || status=90
    fi

    case "$local_dir" in
        /scratch/"$USER"/"$SLURM_JOB_ID") rm -rf -- "$local_dir" ;;
        *) echo "refusing unsafe cleanup path: $local_dir" >&2 ;;
    esac
    exit "$status"
}
trap cleanup EXIT TERM INT

{
    echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "job_id=$SLURM_JOB_ID"
    echo "job_name=$SLURM_JOB_NAME"
    echo "host=$(hostname -f 2>/dev/null || hostname)"
    echo "submit_dir=$submit_dir"
    echo "run_dir=$run_dir"
    echo "local_dir=$local_dir"
    echo "git_commit=$(git -C "$submit_dir" rev-parse HEAD 2>/dev/null || true)"
    echo "cpus_per_task=${SLURM_CPUS_PER_TASK-}"
    echo "mem_per_node=${SLURM_MEM_PER_NODE-}"
    echo
    echo "[modules]"
    module -t list 2>&1 || true
    echo
    echo "[slurm environment]"
    env | grep '^SLURM_' | sort
} >"$run_dir/manifest.txt"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$local_dir"

if [[ $# -gt 0 ]]; then
    printf 'executing:' | tee "$run_dir/command.txt"
    printf ' %q' "$@" | tee -a "$run_dir/command.txt"
    printf '\n' | tee -a "$run_dir/command.txt"
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$@"
else
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -lc '
        set -euo pipefail
        echo "CPU smoke test"
        hostname
        echo "visible CPUs: $(nproc)"
        echo "OMP_NUM_THREADS=${OMP_NUM_THREADS-}"
        printf "completed=%s\n" "$(date --iso-8601=seconds 2>/dev/null || date)" \
          > output/result.txt
    '
fi
