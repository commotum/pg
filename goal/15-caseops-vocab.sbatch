#!/usr/bin/env bash
#SBATCH --job-name=pg-p15-caseops
#SBATCH --partition=share
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/phase15-caseops-vocab/slurm-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/phase15-caseops-vocab/slurm-%j.err

set -euo pipefail
umask 077

VOCAB_SIZE=${VOCAB_SIZE:?VOCAB_SIZE is required, e.g. VOCAB_SIZE=12288}
RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/phase15-caseops-vocab}
ENV_PATH=${ENV_PATH:-/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
PG_PATH=${PG_PATH:-/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf}
RECORD_NAME=${RECORD_NAME:-2026-04-23_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE}
RECORD_PATH=${RECORD_PATH:-$PG_PATH/records/track_10min_16mb/$RECORD_NAME}
CASEOPS_ROOT=${CASEOPS_ROOT:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp${VOCAB_SIZE}}
CASEOPS_OUT=${CASEOPS_OUT:-$CASEOPS_ROOT/datasets}
DOCS_SELECTED=${DOCS_SELECTED:-/nfs/hpc/share/peterj29/pg/data-exports/sp8192-80/docs_selected.jsonl}
TOKENIZER_TRAIN_DOCS=${TOKENIZER_TRAIN_DOCS:-500000}
TOKENIZER_PATH=${TOKENIZER_PATH:-$CASEOPS_OUT/tokenizers/fineweb_${VOCAB_SIZE}_bpe_lossless_caps_caseops_v1_reserved.model}
DATASET_NAME=${DATASET_NAME:-fineweb10B_sp${VOCAB_SIZE}_lossless_caps_caseops_v1_reserved}
DATA_PATH=${DATA_PATH:-$CASEOPS_OUT/datasets/$DATASET_NAME}
VAL_DOCS=${VAL_DOCS:-10000}
FORCE_REBUILD_TOKENIZER=${FORCE_REBUILD_TOKENIZER:-0}
FORCE_REBUILD_DATA=${FORCE_REBUILD_DATA:-0}
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
if [[ ! -f "$REPO_PATH/goal/15-train-caseops-tokenizer.py" ]]; then
    echo "missing tokenizer helper under REPO_PATH=$REPO_PATH" >&2
    exit 2
fi
if [[ ! -f "$RECORD_PATH/prepare_caseops_data.py" ]]; then
    echo "missing prepare_caseops_data.py under RECORD_PATH=$RECORD_PATH" >&2
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
    echo "repo_path=$REPO_PATH"
    echo "parameter_golf_path=$PG_PATH"
    echo "record_path=$RECORD_PATH"
    echo "vocab_size=$VOCAB_SIZE"
    echo "caseops_root=$CASEOPS_ROOT"
    echo "caseops_out=$CASEOPS_OUT"
    echo "docs_selected=$DOCS_SELECTED"
    echo "tokenizer_train_docs=$TOKENIZER_TRAIN_DOCS"
    echo "tokenizer_path=$TOKENIZER_PATH"
    echo "dataset_name=$DATASET_NAME"
    echo "data_path=$DATA_PATH"
    echo "val_docs=$VAL_DOCS"
    echo "force_rebuild_tokenizer=$FORCE_REBUILD_TOKENIZER"
    echo "force_rebuild_data=$FORCE_REBUILD_DATA"
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
    echo "VOCAB_SIZE=$VOCAB_SIZE"
    echo "DOCS_SELECTED=$DOCS_SELECTED"
    echo "TOKENIZER_PATH=$TOKENIZER_PATH"
    echo "TOKENIZER_TRAIN_DOCS=$TOKENIZER_TRAIN_DOCS"
    echo "DATASET_NAME=$DATASET_NAME"
    echo "DATA_PATH=$DATA_PATH"
    echo "CASEOPS_OUT=$CASEOPS_OUT"
    echo "VAL_DOCS=$VAL_DOCS"
    echo "tokenizer_command=python $REPO_PATH/goal/15-train-caseops-tokenizer.py --docs $DOCS_SELECTED --record-path $RECORD_PATH --out-model $TOKENIZER_PATH --vocab-size $VOCAB_SIZE --tokenizer-train-docs $TOKENIZER_TRAIN_DOCS"
    echo "data_command=python $RECORD_PATH/prepare_caseops_data.py --docs $DOCS_SELECTED --out $CASEOPS_OUT --sp $TOKENIZER_PATH --val-docs $VAL_DOCS --dataset-name $DATASET_NAME"
} >"$JOB_DIR/command.txt"

if [[ "$FORCE_REBUILD_TOKENIZER" == "1" || ! -f "$TOKENIZER_PATH" ]]; then
    python "$REPO_PATH/goal/15-train-caseops-tokenizer.py" \
        --docs "$DOCS_SELECTED" \
        --record-path "$RECORD_PATH" \
        --out-model "$TOKENIZER_PATH" \
        --vocab-size "$VOCAB_SIZE" \
        --tokenizer-train-docs "$TOKENIZER_TRAIN_DOCS" \
        >"$JOB_DIR/tokenizer.log" 2>&1
else
    echo "tokenizer exists, skipping: $TOKENIZER_PATH" >"$JOB_DIR/tokenizer.log"
fi

train_shards_existing=$(find "$DATA_PATH" -name 'fineweb_train_*.bin' 2>/dev/null | wc -l)
val_shards_existing=$(find "$DATA_PATH" -name 'fineweb_val_*.bin' 2>/dev/null | wc -l)
val_byte_shards_existing=$(find "$DATA_PATH" -name 'fineweb_val_bytes_*.bin' 2>/dev/null | wc -l)
if [[ "$FORCE_REBUILD_DATA" == "1" || "$train_shards_existing" -lt 1 || "$val_shards_existing" -lt 1 || "$val_byte_shards_existing" -lt 1 ]]; then
    python "$RECORD_PATH/prepare_caseops_data.py" \
        --docs "$DOCS_SELECTED" \
        --out "$CASEOPS_OUT" \
        --sp "$TOKENIZER_PATH" \
        --val-docs "$VAL_DOCS" \
        --dataset-name "$DATASET_NAME" \
        >"$JOB_DIR/caseops-data.log" 2>&1
else
    echo "data exists, skipping: $DATA_PATH" >"$JOB_DIR/caseops-data.log"
fi

{
    echo "[caseops root]"
    find "$CASEOPS_ROOT" -maxdepth 4 -type f -printf '%P\t%s bytes\n' | sort
    echo
    echo "[shard counts]"
    train_shards=$(find "$DATA_PATH" -name 'fineweb_train_*.bin' 2>/dev/null | wc -l)
    val_shards=$(find "$DATA_PATH" -name 'fineweb_val_*.bin' 2>/dev/null | wc -l)
    val_byte_shards=$(find "$DATA_PATH" -name 'fineweb_val_bytes_*.bin' 2>/dev/null | wc -l)
    echo "train_shards=$train_shards"
    echo "val_shards=$val_shards"
    echo "val_byte_shards=$val_byte_shards"
} >"$JOB_DIR/export-files.txt" 2>&1

echo "complete" >"$JOB_DIR/COMPLETE.txt"
