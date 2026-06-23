#!/usr/bin/env bash
#SBATCH --job-name=pg-phase7-data
#SBATCH --partition=share
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/phase7-sp4096-data/slurm-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/phase7-sp4096-data/slurm-%j.err

set -euo pipefail
umask 077

RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/phase7-sp4096-data}
ENV_PATH=${ENV_PATH:-/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
PG_PATH=${PG_PATH:-/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf}
EXPORT_ROOT=${EXPORT_ROOT:-/nfs/hpc/share/peterj29/pg/data-exports/sp4096-80}
CONFIG_PATH=${CONFIG_PATH:-$RUN_ROOT/sp4096-tokenizer-config.json}
JOB_DIR="$RUN_ROOT/$SLURM_JOB_ID"

mkdir -p "$JOB_DIR" "$EXPORT_ROOT"
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.out" "$JOB_DIR/slurm.out" || true
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.err" "$JOB_DIR/slurm.err" || true

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm/current 2>/dev/null || module load slurm 2>/dev/null || true
fi
source "$ENV_PATH/bin/activate"

export HF_HOME=${HF_HOME:-/nfs/hpc/share/peterj29/pg/hf-cache}
export MATCHED_FINEWEB_TOKENIZER_THREADS=${MATCHED_FINEWEB_TOKENIZER_THREADS:-${SLURM_CPUS_PER_TASK:-8}}
export MATCHED_FINEWEB_SP_BATCH_SIZE=${MATCHED_FINEWEB_SP_BATCH_SIZE:-2048}
export MATCHED_FINEWEB_GPT2_DECODE_BATCH_SIZE=${MATCHED_FINEWEB_GPT2_DECODE_BATCH_SIZE:-512}

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
    echo "export_root=$EXPORT_ROOT"
    echo "config_path=$CONFIG_PATH"
    echo "hf_home=$HF_HOME"
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
    echo "[python]"
    command -v python
    python -VV
    echo
    echo "[pip freeze]"
    python -m pip freeze --all
} >"$JOB_DIR/python.txt" 2>&1

{
    echo "HF_HOME=$HF_HOME"
    echo "MATCHED_FINEWEB_TOKENIZER_THREADS=$MATCHED_FINEWEB_TOKENIZER_THREADS"
    echo "MATCHED_FINEWEB_SP_BATCH_SIZE=$MATCHED_FINEWEB_SP_BATCH_SIZE"
    echo "EXPORT_ROOT=$EXPORT_ROOT"
    echo "CONFIG_PATH=$CONFIG_PATH"
    echo "command=python $PG_PATH/data/download_hf_docs_and_tokenize.py --output-root $EXPORT_ROOT --tokenizer-config $CONFIG_PATH --tokenizer-train-docs 500000 --max-train-shards 80 --skip-byte"
} >"$JOB_DIR/command.txt"

python "$PG_PATH/data/download_hf_docs_and_tokenize.py" \
    --output-root "$EXPORT_ROOT" \
    --tokenizer-config "$CONFIG_PATH" \
    --tokenizer-train-docs 500000 \
    --max-train-shards 80 \
    --skip-byte \
    >"$JOB_DIR/export.log" 2>&1

{
    echo "[export root]"
    find "$EXPORT_ROOT" -maxdepth 3 -type f -printf '%P\t%s bytes\n' | sort
    echo
    echo "[manifest summary]"
    python - "$EXPORT_ROOT/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
for tok in payload.get("tokenizers", []):
    print("tokenizer", tok.get("name"), tok.get("vocab_size"), tok.get("model_path"))
for dataset in payload.get("datasets", []):
    stats = dataset.get("stats") or {}
    print(
        "dataset",
        dataset.get("name"),
        "vocab",
        dataset.get("vocab_size"),
        "train_files",
        stats.get("files_train"),
        "val_files",
        stats.get("files_val"),
        "train_tokens",
        stats.get("tokens_train"),
        "val_tokens",
        stats.get("tokens_val"),
    )
PY
} >"$JOB_DIR/export-files.txt"

echo "complete" >"$JOB_DIR/COMPLETE.txt"
