#!/usr/bin/env bash
#SBATCH --job-name=goal3-env-prep
#SBATCH --partition=share
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=goal-3/logs/%x-%j.out
#SBATCH --error=goal-3/logs/%x-%j.err

set -euo pipefail
umask 077

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm
fi
module load "python/${GOAL3_PYTHON_MODULE:-3.12}"

repo_root=${GOAL3_REPO_ROOT:-/nfs/hpc/share/peterj29/pg/src/pg}
script_dir="$repo_root/goal-3/scripts"
# shellcheck disable=SC1091
source "$script_dir/common.sh"

goal3_init_run_dir
goal3_record_context "$GOAL3_RUN_DIR"

tmp_env="$GOAL3_ENV_DIR.tmp.$SLURM_JOB_ID"
cleanup_tmp_env() {
    status=$?
    trap - EXIT TERM INT
    if [[ "$status" -ne 0 && -n "${tmp_env:-}" && -d "$tmp_env" ]]; then
        case "$tmp_env" in
            "$GOAL3_ENV_DIR".tmp."$SLURM_JOB_ID")
                mv "$tmp_env" "$tmp_env.failed"
                echo "Moved failed tmp env to $tmp_env.failed" \
                    | tee -a "$GOAL3_RUN_DIR/status.txt" >&2
                ;;
            *)
                echo "refusing unsafe tmp env cleanup path: $tmp_env" >&2
                ;;
        esac
    fi
    exit "$status"
}
trap cleanup_tmp_env EXIT TERM INT

if [[ -d "$GOAL3_ENV_DIR" && "${GOAL3_ENV_REBUILD:-0}" != "1" ]]; then
    echo "Environment already exists: $GOAL3_ENV_DIR" | tee "$GOAL3_RUN_DIR/status.txt"
    goal3_activate_env
    goal3_record_python_context "$GOAL3_RUN_DIR"
    exit 0
fi

if [[ -e "$tmp_env" ]]; then
    echo "Refusing to reuse existing tmp env: $tmp_env" >&2
    exit 2
fi
if [[ -e "$tmp_env.failed" ]]; then
    echo "Refusing to overwrite existing failed tmp env: $tmp_env.failed" >&2
    exit 2
fi
if ! command -v lrzip >/dev/null 2>&1; then
    echo "WARNING: lrzip is missing. Python env prep will continue, but record runs need goal-3/prepare-tools.sbatch or another lrzip path." \
        | tee -a "$GOAL3_RUN_DIR/status.txt" >&2
fi

python3 -m venv "$tmp_env"
# shellcheck disable=SC1091
source "$tmp_env/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    "${GOAL3_TORCH_SPEC:-torch==2.9.1+cu128}"
python -m pip install sentencepiece brotli huggingface_hub numpy python-minifier
python -m pip install --no-deps flash_attn_3 \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

goal3_record_python_context "$GOAL3_RUN_DIR"

if [[ -d "$GOAL3_ENV_DIR" ]]; then
    backup="$GOAL3_ENV_DIR.backup.$(date +%Y%m%d-%H%M%S)"
    mv "$GOAL3_ENV_DIR" "$backup"
    echo "Backed up existing env to $backup" | tee -a "$GOAL3_RUN_DIR/status.txt"
fi
mv "$tmp_env" "$GOAL3_ENV_DIR"
echo "Environment ready: $GOAL3_ENV_DIR" | tee -a "$GOAL3_RUN_DIR/status.txt"
trap - EXIT TERM INT
