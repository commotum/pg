#!/usr/bin/env bash
#SBATCH --job-name=pg-p13-env
#SBATCH --partition=share
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/phase13-record-env/slurm-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/phase13-record-env/slurm-%j.err

set -euo pipefail
umask 077

RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/phase13-record-env}
ENV_PATH=${ENV_PATH:-/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
PG_PATH=${PG_PATH:-/nfs/hpc/share/peterj29/pg/src/pg/parameter-golf}
RECORD_NAME=${RECORD_NAME:-2026-04-23_SP8192_CaseOps_SparseGate_QuantGate_Loop45_PhasedTTT_PolarNS_MinLR_FusedCE}
RECORD_PATH=${RECORD_PATH:-$PG_PATH/records/track_10min_16mb/$RECORD_NAME}
CASEOPS_ROOT=${CASEOPS_ROOT:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192}
CASEOPS_OUT=${CASEOPS_OUT:-$CASEOPS_ROOT/datasets}
DATA_PATH=${DATA_PATH:-$CASEOPS_OUT/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}
TOKENIZER_PATH=${TOKENIZER_PATH:-$CASEOPS_OUT/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}
SOURCE_TOKENIZER=${SOURCE_TOKENIZER:-$RECORD_PATH/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}
JOB_DIR="$RUN_ROOT/$SLURM_JOB_ID"

mkdir -p "$JOB_DIR"
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.out" "$JOB_DIR/slurm.out" || true
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.err" "$JOB_DIR/slurm.err" || true

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm/current 2>/dev/null || module load slurm 2>/dev/null || true
fi
module load cuda/13.0 2>/dev/null || module load cuda/12.8 2>/dev/null || true
source "$ENV_PATH/bin/activate"

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
    echo "caseops_root=$CASEOPS_ROOT"
    echo "data_path=$DATA_PATH"
    echo "tokenizer_path=$TOKENIZER_PATH"
    echo "source_tokenizer=$SOURCE_TOKENIZER"
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

{
    echo "[files]"
    for path in "$RECORD_PATH/train_gpt.py" "$RECORD_PATH/prepare_caseops_data.py" "$SOURCE_TOKENIZER" "$TOKENIZER_PATH" "$DATA_PATH"; do
        if [[ -e "$path" ]]; then
            printf 'present\t%s\n' "$path"
        else
            printf 'missing\t%s\n' "$path"
        fi
    done
    echo
    echo "[caseops shard counts]"
    train_shards=$(find "$DATA_PATH" -name 'fineweb_train_*.bin' 2>/dev/null | wc -l)
    val_shards=$(find "$DATA_PATH" -name 'fineweb_val_*.bin' ! -name 'fineweb_val_bytes_*.bin' 2>/dev/null | wc -l)
    val_byte_shards=$(find "$DATA_PATH" -name 'fineweb_val_bytes_*.bin' 2>/dev/null | wc -l)
    echo "train_shards=$train_shards"
    echo "val_shards=$val_shards"
    echo "val_byte_shards=$val_byte_shards"
} >"$JOB_DIR/files.txt" 2>&1

{
    echo "[imports]"
    python - <<'PY'
import importlib
import pathlib
import shutil
import subprocess
import sys

mods = [
    "torch",
    "triton",
    "sentencepiece",
    "numpy",
    "flash_attn_interface",
]
for name in mods:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"import_ok {name} version={version}")
    except Exception as exc:
        print(f"import_fail {name} {type(exc).__name__}: {exc}")
        raise

import torch
print(f"torch_cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch_cuda_device_count={torch.cuda.device_count()}")
    print(f"torch_cuda_device_name={torch.cuda.get_device_name(0)}")
    print(f"torch_cuda_capability={torch.cuda.get_device_capability(0)}")
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"cuda_matmul_ok mean={float(y.mean().cpu()):.6f}")

for exe in ["lrzip", "torchrun"]:
    resolved = shutil.which(exe)
    print(f"exe {exe} {resolved or 'missing'}")
    if resolved and exe == "torchrun":
        out = subprocess.run([resolved, "--version"], capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip())

print(f"python={sys.version}")
print(f"cwd={pathlib.Path.cwd()}")
PY
} >"$JOB_DIR/imports.txt" 2>&1

echo "complete" >"$JOB_DIR/COMPLETE.txt"
