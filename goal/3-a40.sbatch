#!/usr/bin/env bash
#SBATCH --job-name=pg-phase3-a40
#SBATCH --partition=share
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/phase3-a40/slurm-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/phase3-a40/slurm-%j.err

set -euo pipefail
umask 077

RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/phase3-a40}
ENV_PATH=${ENV_PATH:-/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
PG_PATH=${PG_PATH:-/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf}
JOB_DIR="$RUN_ROOT/$SLURM_JOB_ID"
WORK_DIR="$JOB_DIR/work"
RUN_ID="baseline_a40_10m_seed42_job${SLURM_JOB_ID}"

mkdir -p "$JOB_DIR" "$WORK_DIR"
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.out" "$JOB_DIR/slurm.out" || true
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.err" "$JOB_DIR/slurm.err" || true

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm/current 2>/dev/null || module load slurm 2>/dev/null || true
fi
module load cuda/13.0 2>/dev/null || module load cuda/12.8 2>/dev/null || true
source "$ENV_PATH/bin/activate"

train_shards=$(find "$PG_PATH/data/datasets/fineweb10B_sp1024" -name 'fineweb_train_*.bin' | wc -l)
val_shards=$(find "$PG_PATH/data/datasets/fineweb10B_sp1024" -name 'fineweb_val_*.bin' | wc -l)
if [[ "$train_shards" -lt 80 || "$val_shards" -lt 1 ]]; then
    echo "missing data: train_shards=$train_shards val_shards=$val_shards" >&2
    exit 2
fi

{
    echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "job_id=$SLURM_JOB_ID"
    echo "job_name=${SLURM_JOB_NAME-}"
    echo "host=$(hostname -f 2>/dev/null || hostname)"
    echo "partition=${SLURM_JOB_PARTITION-}"
    echo "nodes=${SLURM_JOB_NODELIST-}"
    echo "cpus_per_task=${SLURM_CPUS_PER_TASK-}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES-}"
    echo "repo_path=$REPO_PATH"
    echo "parameter_golf_path=$PG_PATH"
    echo "work_dir=$WORK_DIR"
    echo "run_id=$RUN_ID"
    echo "train_shards=$train_shards"
    echo "val_shards=$val_shards"
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

{
    echo "[nvidia-smi -L]"
    nvidia-smi -L
    echo
    echo "[query]"
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
    echo
    echo "[full]"
    nvidia-smi
    echo
    echo "[nvcc]"
    nvcc --version 2>/dev/null || true
} >"$JOB_DIR/gpu.txt" 2>&1

{
    echo "[python]"
    command -v python
    python -VV
    echo
    echo "[pip freeze]"
    python -m pip freeze --all
} >"$JOB_DIR/python.txt" 2>&1

cd "$WORK_DIR"

{
    echo "RUN_ID=$RUN_ID"
    echo "DATA_PATH=$PG_PATH/data/datasets/fineweb10B_sp1024"
    echo "TOKENIZER_PATH=$PG_PATH/data/tokenizers/fineweb_1024_bpe.model"
    echo "VOCAB_SIZE=1024"
    echo "SEED=42"
    echo "ITERATIONS=20000"
    echo "WARMUP_STEPS=20"
    echo "TRAIN_LOG_EVERY=20"
    echo "VAL_LOSS_EVERY=0"
    echo "TRAIN_BATCH_TOKENS=524288"
    echo "VAL_BATCH_SIZE=524288"
    echo "MAX_WALLCLOCK_SECONDS=600"
    echo "command=python $PG_PATH/train_gpt.py"
} >"$JOB_DIR/command.txt"

RUN_ID="$RUN_ID" \
DATA_PATH="$PG_PATH/data/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="$PG_PATH/data/tokenizers/fineweb_1024_bpe.model" \
VOCAB_SIZE=1024 \
SEED=42 \
ITERATIONS=20000 \
WARMUP_STEPS=20 \
TRAIN_LOG_EVERY=20 \
VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=524288 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=600 \
python "$PG_PATH/train_gpt.py" >"$JOB_DIR/train.log" 2>&1

{
    find "$WORK_DIR" -maxdepth 3 -type f -printf '%P\t%s bytes\n' | sort
} >"$JOB_DIR/work-files.txt"

echo "complete" >"$JOB_DIR/COMPLETE.txt"
