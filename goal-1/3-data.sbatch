#!/usr/bin/env bash
#SBATCH --job-name=pg-phase3-data
#SBATCH --partition=share
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/phase3-data/slurm-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/phase3-data/slurm-%j.err

set -euo pipefail
umask 077

RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/phase3-data}
ENV_PATH=${ENV_PATH:-/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
PG_PATH=${PG_PATH:-/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf}
HF_HOME=${HF_HOME:-/nfs/hpc/share/peterj29/pg/hf-cache}
JOB_DIR="$RUN_ROOT/$SLURM_JOB_ID"

mkdir -p "$JOB_DIR" "$HF_HOME"
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.out" "$JOB_DIR/slurm.out" || true
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.err" "$JOB_DIR/slurm.err" || true

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm/current 2>/dev/null || module load slurm 2>/dev/null || true
fi
module load python/3.11
source "$ENV_PATH/bin/activate"
export HF_HOME

{
    echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "job_id=$SLURM_JOB_ID"
    echo "job_name=${SLURM_JOB_NAME-}"
    echo "host=$(hostname -f 2>/dev/null || hostname)"
    echo "partition=${SLURM_JOB_PARTITION-}"
    echo "nodes=${SLURM_JOB_NODELIST-}"
    echo "cpus_per_task=${SLURM_CPUS_PER_TASK-}"
    echo "repo_path=$REPO_PATH"
    echo "parameter_golf_path=$PG_PATH"
    echo "hf_home=$HF_HOME"
    echo "git_commit=$(git -C "$REPO_PATH" rev-parse HEAD 2>/dev/null || true)"
    echo "parameter_golf_commit=$(git -C "$PG_PATH" rev-parse HEAD 2>/dev/null || true)"
    echo
    echo "[modules]"
    module -t list 2>&1 || true
    echo
    echo "[python]"
    command -v python
    python -VV
    echo
    echo "[slurm]"
    env | grep '^SLURM_' | sort
} >"$JOB_DIR/manifest.txt" 2>&1

cd "$PG_PATH"

{
    echo "before_train_shards=$(find data/datasets/fineweb10B_sp1024 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l)"
    echo "before_val_shards=$(find data/datasets/fineweb10B_sp1024 -name 'fineweb_val_*.bin' 2>/dev/null | wc -l)"
    echo "command=python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80"
    python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
    echo "after_train_shards=$(find data/datasets/fineweb10B_sp1024 -name 'fineweb_train_*.bin' | wc -l)"
    echo "after_val_shards=$(find data/datasets/fineweb10B_sp1024 -name 'fineweb_val_*.bin' | wc -l)"
} >"$JOB_DIR/download.log" 2>&1

{
    find data/datasets/fineweb10B_sp1024 -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
    find data/tokenizers -maxdepth 1 -type f -name 'fineweb_1024_bpe*' -printf '%p\t%s bytes\n' | sort
    find data -maxdepth 1 -type f -name 'manifest.json' -printf '%p\t%s bytes\n'
} >"$JOB_DIR/data-files.txt"

echo "complete" >"$JOB_DIR/COMPLETE.txt"
