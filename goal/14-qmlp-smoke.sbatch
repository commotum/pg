#!/usr/bin/env bash
#SBATCH --job-name=pg-p14-qsmoke
#SBATCH --partition=share
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/phase14-record-qmlp-smoke/slurm-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/phase14-record-qmlp-smoke/slurm-%j.err

set -euo pipefail
umask 077

RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/phase14-record-qmlp-smoke}
ENV_PATH=${ENV_PATH:-/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
PG_PATH=${PG_PATH:-/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf}
RECORD_NAME=${RECORD_NAME:-2026-04-23_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE}
RECORD_PATH=${RECORD_PATH:-$PG_PATH/records/track_10min_16mb/$RECORD_NAME}
CASEOPS_ROOT=${CASEOPS_ROOT:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192}
CASEOPS_OUT=${CASEOPS_OUT:-$CASEOPS_ROOT/datasets}
DATA_PATH=${DATA_PATH:-$CASEOPS_OUT/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}
TOKENIZER_PATH=${TOKENIZER_PATH:-$CASEOPS_OUT/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}
SEED_VALUE=${SEED_VALUE:-42}
PREQUANT_ONLY_VALUE=${PREQUANT_ONLY_VALUE:-${PREQUANT_ONLY:-1}}
GPTQ_RESERVE_SECONDS_VALUE=${GPTQ_RESERVE_SECONDS_VALUE:-${GPTQ_RESERVE_SECONDS:-0.5}}
GPTQ_CALIBRATION_BATCHES_VALUE=${GPTQ_CALIBRATION_BATCHES_VALUE:-${GPTQ_CALIBRATION_BATCHES:-16}}
MAX_WALLCLOCK_SECONDS_VALUE=${MAX_WALLCLOCK_SECONDS_VALUE:-${MAX_WALLCLOCK_SECONDS:-60}}
JOB_DIR="$RUN_ROOT/$SLURM_JOB_ID"
WORK_DIR="$JOB_DIR/work"
ARTIFACT_DIR="$WORK_DIR/artifact"
RUN_ID="record0423_qmlp_a40_smoke_seed${SEED_VALUE}_job${SLURM_JOB_ID}"

mkdir -p "$JOB_DIR" "$WORK_DIR" "$ARTIFACT_DIR"
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.out" "$JOB_DIR/slurm.out" || true
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.err" "$JOB_DIR/slurm.err" || true

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm/current 2>/dev/null || module load slurm 2>/dev/null || true
fi
module load cuda/13.0 2>/dev/null || module load cuda/12.8 2>/dev/null || true
source "$ENV_PATH/bin/activate"

if [[ ! -f "$RECORD_PATH/train_gpt.py" ]]; then
    echo "missing record train_gpt.py under RECORD_PATH=$RECORD_PATH" >&2
    exit 2
fi
if [[ ! -f "$TOKENIZER_PATH" ]]; then
    echo "missing TOKENIZER_PATH=$TOKENIZER_PATH" >&2
    exit 2
fi

train_shards=$(find "$DATA_PATH" -name 'fineweb_train_*.bin' 2>/dev/null | wc -l)
val_shards=$(find "$DATA_PATH" -name 'fineweb_val_*.bin' ! -name 'fineweb_val_bytes_*.bin' 2>/dev/null | wc -l)
val_byte_shards=$(find "$DATA_PATH" -name 'fineweb_val_bytes_*.bin' 2>/dev/null | wc -l)
if [[ "$train_shards" -lt 1 || "$val_shards" -lt 1 || "$val_byte_shards" -lt 1 ]]; then
    echo "missing CaseOps data: train_shards=$train_shards val_shards=$val_shards val_byte_shards=$val_byte_shards" >&2
    exit 2
fi

{
    echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "job_id=$SLURM_JOB_ID"
    echo "job_name=${SLURM_JOB_NAME-}"
    echo "host=$(hostname -f 2>/dev/null || hostname)"
    echo "partition=${SLURM_JOB_PARTITION-}"
    echo "nodes=${SLURM_JOB_NODELIST-}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES-}"
    echo "repo_path=$REPO_PATH"
    echo "parameter_golf_path=$PG_PATH"
    echo "record_path=$RECORD_PATH"
    echo "data_path=$DATA_PATH"
    echo "tokenizer_path=$TOKENIZER_PATH"
    echo "artifact_dir=$ARTIFACT_DIR"
    echo "run_id=$RUN_ID"
    echo "seed_value=$SEED_VALUE"
    echo "train_shards=$train_shards"
    echo "val_shards=$val_shards"
    echo "val_byte_shards=$val_byte_shards"
    echo "quat_mlp=1"
    echo "quat_mlp_impl=matrix"
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
    echo "[full]"
    nvidia-smi
    echo
    echo "[nvcc]"
    nvcc --version 2>/dev/null || true
} >"$JOB_DIR/gpu.txt" 2>&1

cd "$WORK_DIR"

{
    echo "NCCL_NET=Socket"
    echo "DATA_PATH=$DATA_PATH"
    echo "TOKENIZER_PATH=$TOKENIZER_PATH"
    echo "ARTIFACT_DIR=$ARTIFACT_DIR"
    echo "CASEOPS_ENABLED=1"
    echo "QUAT_MLP=1"
    echo "QUAT_MLP_IMPL=matrix"
    echo "DOCUMENT_PACKING=0"
    echo "TORCH_COMPILE=0"
    echo "FUSED_MLP_ENABLED=0"
    echo "TTT_ENABLED=0"
    echo "PREQUANT_ONLY=$PREQUANT_ONLY_VALUE"
    echo "SEED=$SEED_VALUE"
    echo "ITERATIONS=2"
    echo "WARMUP_STEPS=1"
    echo "TRAIN_LOG_EVERY=1"
    echo "VAL_LOSS_EVERY=0"
    echo "TRAIN_BATCH_TOKENS=65536"
    echo "VAL_BATCH_TOKENS=65536"
    echo "GPTQ_RESERVE_SECONDS=$GPTQ_RESERVE_SECONDS_VALUE"
    echo "GPTQ_CALIBRATION_BATCHES=$GPTQ_CALIBRATION_BATCHES_VALUE"
    echo "MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS_VALUE"
    echo "command=torchrun --standalone --nproc_per_node=1 $RECORD_PATH/train_gpt.py"
} >"$JOB_DIR/command.txt"

NCCL_NET=Socket \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
ARTIFACT_DIR="$ARTIFACT_DIR" \
RUN_ID="$RUN_ID" \
CASEOPS_ENABLED=1 \
QUAT_MLP=1 \
QUAT_MLP_IMPL=matrix \
DOCUMENT_PACKING=0 \
TORCH_COMPILE=0 \
FUSED_MLP_ENABLED=0 \
TTT_ENABLED=0 \
PREQUANT_ONLY="$PREQUANT_ONLY_VALUE" \
PHASED_TTT_PREFIX_DOCS=2000 \
PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 \
ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 \
EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS="$GPTQ_RESERVE_SECONDS_VALUE" \
GPTQ_CALIBRATION_BATCHES="$GPTQ_CALIBRATION_BATCHES_VALUE" \
VAL_LOSS_EVERY=0 \
MIN_LR=0.10 \
FUSED_CE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_INIT_STD=0.0 \
SPARSE_ATTN_GATE_SCALE=1.0 \
GATED_ATTN_ENABLED=0 \
ATTN_OUT_GATE_ENABLED=0 \
GATED_ATTN_QUANT_GATE=1 \
SEED="$SEED_VALUE" \
ITERATIONS=2 \
WARMUP_STEPS=1 \
TRAIN_LOG_EVERY=1 \
TRAIN_BATCH_TOKENS=65536 \
VAL_BATCH_TOKENS=65536 \
MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS_VALUE" \
torchrun --standalone --nproc_per_node=1 "$RECORD_PATH/train_gpt.py" >"$JOB_DIR/train.log" 2>&1

{
    find "$WORK_DIR" -maxdepth 4 -type f -printf '%P\t%s bytes\n' | sort
} >"$JOB_DIR/work-files.txt"

echo "complete" >"$JOB_DIR/COMPLETE.txt"
