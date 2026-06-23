#!/usr/bin/env bash
#SBATCH --job-name=pg-p13-ttt
#SBATCH --partition=share
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/phase13-record-ttt-eval/slurm-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/phase13-record-ttt-eval/slurm-%j.err

set -euo pipefail
umask 077

ARTIFACT_DIR=${ARTIFACT_DIR:?ARTIFACT_DIR with final_model.int6.ptz is required}
RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/phase13-record-ttt-eval}
ENV_PATH=${ENV_PATH:-/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
PG_PATH=${PG_PATH:-/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf}
RECORD_NAME=${RECORD_NAME:-2026-04-23_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE}
RECORD_PATH=${RECORD_PATH:-$PG_PATH/records/track_10min_16mb/$RECORD_NAME}
CASEOPS_ROOT=${CASEOPS_ROOT:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched}
CASEOPS_OUT=${CASEOPS_OUT:-$CASEOPS_ROOT/datasets}
DATA_PATH=${DATA_PATH:-$CASEOPS_OUT/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}
TOKENIZER_PATH=${TOKENIZER_PATH:-$CASEOPS_OUT/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}
SEED_VALUE=${SEED_VALUE:-42}
TTT_BATCH_SIZE_VALUE=${TTT_BATCH_SIZE_VALUE:-${TTT_BATCH_SIZE:-32}}
JOB_DIR="$RUN_ROOT/$SLURM_JOB_ID"
RUN_ID="record0423_a40_ttt_eval_seed${SEED_VALUE}_bs${TTT_BATCH_SIZE_VALUE}_job${SLURM_JOB_ID}"

mkdir -p "$JOB_DIR"
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.out" "$JOB_DIR/slurm.out" || true
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.err" "$JOB_DIR/slurm.err" || true

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm/current 2>/dev/null || module load slurm 2>/dev/null || true
fi
module load cuda/13.0 2>/dev/null || module load cuda/12.8 2>/dev/null || true
source "$ENV_PATH/bin/activate"

if [[ ! -f "$ARTIFACT_DIR/final_model.int6.ptz" ]]; then
    echo "missing quantized artifact: $ARTIFACT_DIR/final_model.int6.ptz" >&2
    exit 2
fi

{
    echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "job_id=$SLURM_JOB_ID"
    echo "artifact_dir=$ARTIFACT_DIR"
    echo "data_path=$DATA_PATH"
    echo "tokenizer_path=$TOKENIZER_PATH"
    echo "seed_value=$SEED_VALUE"
    echo "ttt_batch_size=$TTT_BATCH_SIZE_VALUE"
    echo "run_id=$RUN_ID"
    echo "parameter_golf_status=$(git -C "$PG_PATH" status --short 2>/dev/null | tr '\n' ';')"
} >"$JOB_DIR/manifest.txt" 2>&1

{
    echo "TTT_EVAL_ONLY=1"
    echo "ARTIFACT_DIR=$ARTIFACT_DIR"
    echo "DATA_PATH=$DATA_PATH"
    echo "TOKENIZER_PATH=$TOKENIZER_PATH"
    echo "TTT_BATCH_SIZE=$TTT_BATCH_SIZE_VALUE"
    echo "command=torchrun --standalone --nproc_per_node=1 $RECORD_PATH/train_gpt.py"
} >"$JOB_DIR/command.txt"

NCCL_NET=Socket \
DATA_PATH="$DATA_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
ARTIFACT_DIR="$ARTIFACT_DIR" \
RUN_ID="$RUN_ID" \
CASEOPS_ENABLED=1 \
DOCUMENT_PACKING=0 \
TORCH_COMPILE=0 \
FUSED_MLP_ENABLED=0 \
TTT_ENABLED=1 \
TTT_EVAL_ONLY=1 \
TTT_BATCH_SIZE="$TTT_BATCH_SIZE_VALUE" \
PHASED_TTT_PREFIX_DOCS=2000 \
PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 \
ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 \
EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
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
TRAIN_BATCH_TOKENS=262144 \
VAL_BATCH_TOKENS=262144 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 "$RECORD_PATH/train_gpt.py" >"$JOB_DIR/train.log" 2>&1

find "$ARTIFACT_DIR" -maxdepth 2 -type f -printf '%P\t%s bytes\n' | sort >"$JOB_DIR/artifact-files.txt"
echo "complete" >"$JOB_DIR/COMPLETE.txt"
