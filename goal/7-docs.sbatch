#!/usr/bin/env bash
#SBATCH --job-name=pg-phase7-docs
#SBATCH --partition=share
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/phase7-docs/slurm-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/phase7-docs/slurm-%j.err

set -euo pipefail
umask 077

RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/phase7-docs}
ENV_PATH=${ENV_PATH:-/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
PG_PATH=${PG_PATH:-/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf}
JOB_DIR="$RUN_ROOT/$SLURM_JOB_ID"

mkdir -p "$JOB_DIR"
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.out" "$JOB_DIR/slurm.out" || true
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.err" "$JOB_DIR/slurm.err" || true

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm/current 2>/dev/null || module load slurm 2>/dev/null || true
fi
source "$ENV_PATH/bin/activate"

export HF_HOME=${HF_HOME:-/nfs/hpc/share/peterj29/pg/hf-cache}

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
    echo "parameter_golf_status=$(git -C "$PG_PATH" status --short 2>/dev/null | tr '\n' ';')"
    echo
    echo "[python]"
    command -v python
    python -VV
    echo
    echo "[slurm]"
    env | grep '^SLURM_' | sort
} >"$JOB_DIR/manifest.txt" 2>&1

{
    echo "HF_HOME=$HF_HOME"
    echo "command=python $PG_PATH/data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0 --with-docs"
} >"$JOB_DIR/command.txt"

cd "$PG_PATH"
python "$PG_PATH/data/cached_challenge_fineweb.py" \
    --variant sp1024 \
    --train-shards 0 \
    --with-docs \
    >"$JOB_DIR/download.log" 2>&1

{
    ls -lh "$PG_PATH/data/docs_selected.jsonl" "$PG_PATH/data/docs_selected.source_manifest.json"
    sha256sum "$PG_PATH/data/docs_selected.jsonl" "$PG_PATH/data/docs_selected.source_manifest.json"
    du -sh "$HF_HOME"
} >"$JOB_DIR/docs-files.txt"

echo "complete" >"$JOB_DIR/COMPLETE.txt"
