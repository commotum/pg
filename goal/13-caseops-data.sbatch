#!/usr/bin/env bash
#SBATCH --job-name=pg-p13-caseops
#SBATCH --partition=share
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/phase13-caseops-data/slurm-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/phase13-caseops-data/slurm-%j.err

set -euo pipefail
umask 077

RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/phase13-caseops-data}
ENV_PATH=${ENV_PATH:-/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
PG_PATH=${PG_PATH:-/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf}
RECORD_NAME=${RECORD_NAME:-2026-04-23_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE}
RECORD_PATH=${RECORD_PATH:-$PG_PATH/records/track_10min_16mb/$RECORD_NAME}
CASEOPS_ROOT=${CASEOPS_ROOT:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192}
CASEOPS_OUT=${CASEOPS_OUT:-$CASEOPS_ROOT/datasets}
DOCS_SELECTED=${DOCS_SELECTED:-/nfs/hpc/share/peterj29/pg/data-exports/sp8192-80/docs_selected.jsonl}
SOURCE_TOKENIZER=${SOURCE_TOKENIZER:-$RECORD_PATH/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}
TARGET_TOKENIZER=${TARGET_TOKENIZER:-$CASEOPS_OUT/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}
DATA_PATH=${DATA_PATH:-$CASEOPS_OUT/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}
VAL_DOCS=${VAL_DOCS:-10000}
MAX_DOCS=${MAX_DOCS:-0}
MAX_TRAIN_SHARDS=${MAX_TRAIN_SHARDS:-0}
DATASET_NAME=${DATASET_NAME:-fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}
JOB_DIR="$RUN_ROOT/$SLURM_JOB_ID"

mkdir -p "$JOB_DIR" "$CASEOPS_OUT/tokenizers"
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.out" "$JOB_DIR/slurm.out" || true
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.err" "$JOB_DIR/slurm.err" || true

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm/current 2>/dev/null || module load slurm 2>/dev/null || true
fi
source "$ENV_PATH/bin/activate"

if [[ ! -f "$DOCS_SELECTED" ]]; then
    echo "missing DOCS_SELECTED=$DOCS_SELECTED" >&2
    exit 2
fi
if [[ ! -f "$SOURCE_TOKENIZER" ]]; then
    echo "missing SOURCE_TOKENIZER=$SOURCE_TOKENIZER" >&2
    exit 2
fi
if [[ ! -f "$RECORD_PATH/prepare_caseops_data.py" ]]; then
    echo "missing prepare_caseops_data.py under RECORD_PATH=$RECORD_PATH" >&2
    exit 2
fi

cp "$SOURCE_TOKENIZER" "$TARGET_TOKENIZER"

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
    echo "record_path=$RECORD_PATH"
    echo "caseops_root=$CASEOPS_ROOT"
    echo "caseops_out=$CASEOPS_OUT"
    echo "docs_selected=$DOCS_SELECTED"
    echo "source_tokenizer=$SOURCE_TOKENIZER"
    echo "target_tokenizer=$TARGET_TOKENIZER"
    echo "data_path=$DATA_PATH"
    echo "val_docs=$VAL_DOCS"
    echo "max_docs=$MAX_DOCS"
    echo "max_train_shards=$MAX_TRAIN_SHARDS"
    echo "dataset_name=$DATASET_NAME"
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
    echo "DOCS_SELECTED=$DOCS_SELECTED"
    echo "CASEOPS_OUT=$CASEOPS_OUT"
    echo "SOURCE_TOKENIZER=$SOURCE_TOKENIZER"
    echo "VAL_DOCS=$VAL_DOCS"
    echo "MAX_DOCS=$MAX_DOCS"
    echo "MAX_TRAIN_SHARDS=$MAX_TRAIN_SHARDS"
    echo "DATASET_NAME=$DATASET_NAME"
    echo "command=python $RECORD_PATH/prepare_caseops_data.py --docs $DOCS_SELECTED --out $CASEOPS_OUT --sp $SOURCE_TOKENIZER --val-docs $VAL_DOCS --max-docs $MAX_DOCS --max-train-shards $MAX_TRAIN_SHARDS --dataset-name $DATASET_NAME"
} >"$JOB_DIR/command.txt"

python "$RECORD_PATH/prepare_caseops_data.py" \
    --docs "$DOCS_SELECTED" \
    --out "$CASEOPS_OUT" \
    --sp "$SOURCE_TOKENIZER" \
    --val-docs "$VAL_DOCS" \
    --max-docs "$MAX_DOCS" \
    --max-train-shards "$MAX_TRAIN_SHARDS" \
    --dataset-name "$DATASET_NAME" \
    >"$JOB_DIR/caseops-data.log" 2>&1

{
    echo "[caseops root]"
    find "$CASEOPS_ROOT" -maxdepth 4 -type f -printf '%P\t%s bytes\n' | sort
    echo
    echo "[shard counts]"
    train_shards=$(find "$DATA_PATH" -name 'fineweb_train_*.bin' | wc -l)
    val_shards=$(find "$DATA_PATH" -name 'fineweb_val_*.bin' ! -name 'fineweb_val_bytes_*.bin' | wc -l)
    val_byte_shards=$(find "$DATA_PATH" -name 'fineweb_val_bytes_*.bin' | wc -l)
    echo "train_shards=$train_shards"
    echo "val_shards=$val_shards"
    echo "val_byte_shards=$val_byte_shards"
} >"$JOB_DIR/export-files.txt" 2>&1

echo "complete" >"$JOB_DIR/COMPLETE.txt"
