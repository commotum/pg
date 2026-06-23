#!/usr/bin/env bash
#SBATCH --job-name=pg-p11-smoke16
#SBATCH --partition=share
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/phase11-sp16384-smoke-cpu16/slurm-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/phase11-sp16384-smoke-cpu16/slurm-%j.err

set -euo pipefail
umask 077

RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/phase11-sp16384-smoke-cpu16}
ENV_PATH=${ENV_PATH:-/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
PG_PATH=${PG_PATH:-/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf}
EXPORT_ROOT=${EXPORT_ROOT:-/nfs/hpc/share/peterj29/pg/data-exports/sp16384-80-cpu16}
DATA_PATH=${DATA_PATH:-$EXPORT_ROOT/datasets/fineweb10B_sp16384}
TOKENIZER_PATH=${TOKENIZER_PATH:-$EXPORT_ROOT/tokenizers/fineweb_16384_bpe.model}
JOB_DIR="$RUN_ROOT/$SLURM_JOB_ID"
WORK_DIR="$JOB_DIR/work"
RUN_ID="qmlp_matrix_sp16384_smoke_a40_cpu16_job${SLURM_JOB_ID}"

mkdir -p "$JOB_DIR" "$WORK_DIR"
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.out" "$JOB_DIR/slurm.out" || true
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.err" "$JOB_DIR/slurm.err" || true

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm/current 2>/dev/null || module load slurm 2>/dev/null || true
fi
module load cuda/13.0 2>/dev/null || module load cuda/12.8 2>/dev/null || true
source "$ENV_PATH/bin/activate"

train_shards=$(find "$DATA_PATH" -name 'fineweb_train_*.bin' | wc -l)
val_shards=$(find "$DATA_PATH" -name 'fineweb_val_*.bin' | wc -l)
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
    echo "export_root=$EXPORT_ROOT"
    echo "data_path=$DATA_PATH"
    echo "tokenizer_path=$TOKENIZER_PATH"
    echo "work_dir=$WORK_DIR"
    echo "run_id=$RUN_ID"
    echo "train_shards=$train_shards"
    echo "val_shards=$val_shards"
    echo "git_commit=$(git -C "$REPO_PATH" rev-parse HEAD 2>/dev/null || true)"
    echo "parameter_golf_commit=$(git -C "$PG_PATH" rev-parse HEAD 2>/dev/null || true)"
    echo "parameter_golf_status=$(git -C "$PG_PATH" status --short 2>/dev/null | tr '\n' ';')"
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
    echo "[nvcc]"
    nvcc --version 2>/dev/null || true
} >"$JOB_DIR/gpu.txt" 2>&1

cd "$WORK_DIR"

{
    echo "RUN_ID=$RUN_ID"
    echo "QUAT_MLP=1"
    echo "QUAT_MLP_IMPL=matrix"
    echo "DATA_PATH=$DATA_PATH"
    echo "TOKENIZER_PATH=$TOKENIZER_PATH"
    echo "VOCAB_SIZE=16384"
    echo "SEED=42"
    echo "ITERATIONS=2"
    echo "WARMUP_STEPS=1"
    echo "TRAIN_LOG_EVERY=1"
    echo "VAL_LOSS_EVERY=0"
    echo "TRAIN_BATCH_TOKENS=65536"
    echo "VAL_BATCH_SIZE=65536"
    echo "MAX_WALLCLOCK_SECONDS=0"
    echo "command=python $PG_PATH/train_gpt.py"
} >"$JOB_DIR/command.txt"

RUN_ID="$RUN_ID" \
QUAT_MLP=1 \
QUAT_MLP_IMPL=matrix \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
VOCAB_SIZE=16384 \
SEED=42 \
ITERATIONS=2 \
WARMUP_STEPS=1 \
TRAIN_LOG_EVERY=1 \
VAL_LOSS_EVERY=0 \
TRAIN_BATCH_TOKENS=65536 \
VAL_BATCH_SIZE=65536 \
MAX_WALLCLOCK_SECONDS=0 \
python "$PG_PATH/train_gpt.py" >"$JOB_DIR/train.log" 2>&1

{
    find "$WORK_DIR" -maxdepth 3 -type f -printf '%P\t%s bytes\n' | sort
} >"$JOB_DIR/work-files.txt"

echo "complete" >"$JOB_DIR/COMPLETE.txt"
