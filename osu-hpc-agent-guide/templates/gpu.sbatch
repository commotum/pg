#!/usr/bin/env bash
#SBATCH --job-name=osu-gpu-smoke
#SBATCH --partition=dgxh
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=osu-gpu-%j.out
#SBATCH --error=osu-gpu-%j.err

# Before submission, verify the live partition, account, GRES, and feature:
#   sinfo -p dgxh -N -o '%N|%T|%G|%f'
#
# Submit as-is for a short nvidia-smi smoke test:
#   sbatch gpu.sbatch
#
# Or pass a command:
#   sbatch gpu.sbatch python train.py --config config/small.yaml
#
# Change `--constraint` to a live supported model/VRAM feature when needed.
# A constraint does NOT allocate a GPU; keep `--gres=gpu:1`.

set -euo pipefail
umask 077

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm
fi
# module load cuda/VERSION
# module load python/VERSION
# source /nfs/hpc/share/$USER/envs/PROJECT/bin/activate

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
    echo "host=$(hostname -f 2>/dev/null || hostname)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
    echo "git_commit=$(git -C "$submit_dir" rev-parse HEAD 2>/dev/null || true)"
    echo
    echo "[modules]"
    module -t list 2>&1 || true
    echo
    echo "[slurm environment]"
    env | grep '^SLURM_' | sort
} >"$run_dir/manifest.txt"

{
    nvidia-smi -L
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
    nvidia-smi
    nvcc --version 2>/dev/null || true
} >"$run_dir/gpu.txt" 2>&1

cd "$local_dir"

if [[ $# -gt 0 ]]; then
    printf 'executing:' | tee "$run_dir/command.txt"
    printf ' %q' "$@" | tee -a "$run_dir/command.txt"
    printf '\n' | tee -a "$run_dir/command.txt"
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" "$@"
else
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -lc '
        set -euo pipefail
        nvidia-smi -L
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
        printf "completed=%s\n" "$(date --iso-8601=seconds 2>/dev/null || date)" \
          > output/result.txt
    '
fi
