#!/usr/bin/env bash

set -euo pipefail

GOAL3_REPO_ROOT=${GOAL3_REPO_ROOT:-/nfs/hpc/share/peterj29/pg/src/pg}
GOAL3_STAGE_DIR=${GOAL3_STAGE_DIR:-$GOAL3_REPO_ROOT/goal-3/stage/primary-qmlp}
GOAL3_RUN_ROOT=${GOAL3_RUN_ROOT:-/nfs/hpc/share/peterj29/pg/goal-3-runs}
GOAL3_ENV_DIR=${GOAL3_ENV_DIR:-/nfs/hpc/share/peterj29/pg/envs/goal3-cu128}
GOAL3_TOOLS_DIR=${GOAL3_TOOLS_DIR:-/nfs/hpc/share/peterj29/pg/tools}
GOAL3_ARTIFACT_LIMIT=${GOAL3_ARTIFACT_LIMIT:-16000000}
GOAL3_FA3_FIND_LINKS=${GOAL3_FA3_FIND_LINKS:-https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/}

GOAL3_SP8192_DATA=${GOAL3_SP8192_DATA:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}
GOAL3_SP8192_TOKENIZER=${GOAL3_SP8192_TOKENIZER:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}
GOAL3_SP16384_DATA=${GOAL3_SP16384_DATA:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/datasets/datasets/fineweb10B_sp16384_lossless_caps_caseops_v1_reserved}
GOAL3_SP16384_TOKENIZER=${GOAL3_SP16384_TOKENIZER:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/datasets/tokenizers/fineweb_16384_bpe_lossless_caps_caseops_v1_reserved.model}

export GOAL3_REPO_ROOT GOAL3_STAGE_DIR GOAL3_RUN_ROOT GOAL3_ENV_DIR
export GOAL3_TOOLS_DIR
export GOAL3_ARTIFACT_LIMIT
export GOAL3_FA3_FIND_LINKS
export GOAL3_SP8192_DATA GOAL3_SP8192_TOKENIZER
export GOAL3_SP16384_DATA GOAL3_SP16384_TOKENIZER

export PATH="$GOAL3_TOOLS_DIR/lrzip/bin:$GOAL3_TOOLS_DIR/bin:$PATH"

goal3_timestamp() {
    date --iso-8601=seconds 2>/dev/null || date
}

goal3_init_run_dir() {
    local job_name=${SLURM_JOB_NAME:-goal3-local}
    local job_id=${SLURM_JOB_ID:-manual}
    GOAL3_RUN_DIR=${GOAL3_RUN_DIR:-$GOAL3_RUN_ROOT/${job_name}-${job_id}}
    mkdir -p "$GOAL3_RUN_DIR"
    export GOAL3_RUN_DIR
}

goal3_prepare_runtime_storage() {
    local local_root=${GOAL3_LOCAL_ROOT:-/scratch/$USER/${SLURM_JOB_ID:-manual}/goal3}
    local cache_root=${GOAL3_CACHE_ROOT:-$local_root/cache}
    local tmp_root=${GOAL3_TMP_ROOT:-$local_root/tmp}
    local manifest=${GOAL3_RUN_DIR:-$PWD}/runtime-storage.txt

    mkdir -p \
        "$cache_root/xdg" \
        "$cache_root/torchinductor" \
        "$cache_root/triton" \
        "$cache_root/cuda" \
        "$cache_root/pip" \
        "$tmp_root" \
        "$(dirname "$manifest")"

    export GOAL3_LOCAL_ROOT=$local_root
    export GOAL3_CACHE_ROOT=$cache_root
    export GOAL3_TMP_ROOT=$tmp_root
    export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$cache_root/xdg}
    export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$cache_root/torchinductor}
    export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$cache_root/triton}
    export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$cache_root/cuda}
    export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$cache_root/pip}
    export TMPDIR=${TMPDIR:-$tmp_root}
    export TEMP=${TEMP:-$TMPDIR}
    export TMP=${TMP:-$TMPDIR}

    {
        echo "timestamp=$(goal3_timestamp)"
        echo "GOAL3_LOCAL_ROOT=$GOAL3_LOCAL_ROOT"
        echo "GOAL3_CACHE_ROOT=$GOAL3_CACHE_ROOT"
        echo "GOAL3_TMP_ROOT=$GOAL3_TMP_ROOT"
        echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
        echo "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
        echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
        echo "CUDA_CACHE_PATH=$CUDA_CACHE_PATH"
        echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
        echo "TMPDIR=$TMPDIR"
    } >"$manifest"
}

goal3_activate_env() {
    if [[ ! -d "$GOAL3_ENV_DIR" ]]; then
        echo "missing GOAL3_ENV_DIR=$GOAL3_ENV_DIR" >&2
        echo "Run goal-3/prepare-env.sbatch before H100 execution." >&2
        return 2
    fi
    if [[ ! -x "$GOAL3_ENV_DIR/bin/python" ]]; then
        echo "missing executable Python at $GOAL3_ENV_DIR/bin/python" >&2
        echo "Run goal-3/prepare-env.sbatch before H100 execution." >&2
        return 2
    fi

    # Do not source bin/activate here. Python venv activation scripts and
    # console-script shebangs embed absolute paths, and this env is built via a
    # temporary directory before being moved into place.
    export VIRTUAL_ENV="$GOAL3_ENV_DIR"
    export VIRTUAL_ENV_PROMPT="($(basename "$GOAL3_ENV_DIR")) "
    export PATH="$GOAL3_ENV_DIR/bin:$PATH"
    unset PYTHONHOME
    hash -r 2>/dev/null || true

    local active_python
    active_python=$(command -v python || true)
    if [[ "$active_python" != "$GOAL3_ENV_DIR/bin/python" ]]; then
        echo "failed to activate GOAL3_ENV_DIR=$GOAL3_ENV_DIR; python resolves to ${active_python:-missing}" >&2
        return 2
    fi
    python - "$GOAL3_ENV_DIR" <<'PY'
import os
import sys

expected = sys.argv[1]
actual = sys.prefix
try:
    matches = os.path.samefile(actual, expected)
except OSError:
    matches = os.path.abspath(actual) == os.path.abspath(expected)
if not matches:
    print(f"active Python prefix mismatch: expected {expected}, got {actual}", file=sys.stderr)
    raise SystemExit(2)
PY
}

goal3_repair_venv_metadata() {
    local env_dir=${1:-$GOAL3_ENV_DIR}
    local old_prefix=${2:-}
    if [[ -z "$old_prefix" || ! -d "$env_dir" ]]; then
        return 0
    fi
    local repair_python=$env_dir/bin/python
    if [[ ! -x "$repair_python" ]]; then
        repair_python=$(command -v python3 || command -v python)
    fi
    "$repair_python" - "$env_dir" "$old_prefix" <<'PY'
import stat
import sys
from pathlib import Path

env_dir = Path(sys.argv[1])
old_prefix = sys.argv[2]
old_name = Path(old_prefix).name
new_name = env_dir.name
targets = [env_dir / "pyvenv.cfg"]
bin_dir = env_dir / "bin"
if bin_dir.exists():
    targets.extend(path for path in bin_dir.iterdir() if path.is_file())

for path in targets:
    if not path.exists():
        continue
    try:
        data = path.read_bytes()
    except OSError:
        continue
    if b"\0" in data:
        continue
    text = data.decode("utf-8", errors="surrogateescape")
    patched = text.replace(old_prefix, str(env_dir)).replace(old_name, new_name)
    if patched == text:
        continue
    mode = path.stat().st_mode
    path.write_text(patched, encoding="utf-8", errors="surrogateescape")
    path.chmod(stat.S_IMODE(mode))
PY
}

goal3_record_context() {
    local out_dir=${1:-$GOAL3_RUN_DIR}
    mkdir -p "$out_dir"
    {
        echo "timestamp=$(goal3_timestamp)"
        echo "host=$(hostname -f 2>/dev/null || hostname)"
        echo "pwd=$PWD"
        echo "repo_root=$GOAL3_REPO_ROOT"
        echo "stage_dir=$GOAL3_STAGE_DIR"
        echo "run_dir=$GOAL3_RUN_DIR"
        echo "env_dir=$GOAL3_ENV_DIR"
        echo "local_root=${GOAL3_LOCAL_ROOT-}"
        echo "cache_root=${GOAL3_CACHE_ROOT-}"
        echo "tmp_root=${GOAL3_TMP_ROOT-}"
        echo "torchinductor_cache=${TORCHINDUCTOR_CACHE_DIR-}"
        echo "triton_cache=${TRITON_CACHE_DIR-}"
        echo "cuda_cache=${CUDA_CACHE_PATH-}"
        echo "pip_cache=${PIP_CACHE_DIR-}"
        echo "artifact_limit=$GOAL3_ARTIFACT_LIMIT"
        echo "git_commit=$(git -C "$GOAL3_REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
        echo "git_status_start"
        git -C "$GOAL3_REPO_ROOT" status --short 2>/dev/null || true
        echo "git_status_end"
        echo "submodule_status_start"
        git -C "$GOAL3_REPO_ROOT" submodule status 2>/dev/null || true
        echo "submodule_status_end"
        echo "module_list_start"
        module -t list 2>&1 || true
        echo "module_list_end"
        echo "slurm_env_start"
        env | grep '^SLURM_' | sort || true
        echo "slurm_env_end"
    } >"$out_dir/context.txt"
    git -C "$GOAL3_REPO_ROOT" status --short >"$out_dir/git-status.txt" 2>/dev/null || true
    git -C "$GOAL3_REPO_ROOT" diff --stat >"$out_dir/git-diff.stat" 2>/dev/null || true
    git -C "$GOAL3_REPO_ROOT" diff --submodule=short >"$out_dir/git-diff.patch" 2>/dev/null || true
}

goal3_record_gpu_context() {
    local out_dir=${1:-$GOAL3_RUN_DIR}
    mkdir -p "$out_dir"
    {
        echo "timestamp=$(goal3_timestamp)"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
        nvidia-smi -L
        nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
        nvidia-smi
        nvcc --version 2>/dev/null || true
    } >"$out_dir/gpu.txt" 2>&1
}

goal3_record_python_context() {
    local out_dir=${1:-$GOAL3_RUN_DIR}
    mkdir -p "$out_dir"
    {
        python -VV
        python - <<'PY'
import importlib.util
import shutil

mods = ["torch", "triton", "sentencepiece", "brotli", "flash_attn_interface"]
for name in mods:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {'ok' if spec else 'missing'}")
if importlib.util.find_spec("torch"):
    import torch
    print(f"torch.__version__={torch.__version__}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count={torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"cuda_device_{i}={torch.cuda.get_device_name(i)}")
print(f"lrzip={shutil.which('lrzip')}")
PY
        python -m pip freeze
    } >"$out_dir/python.txt" 2>&1
}

goal3_record_source_snapshot() {
    local out_dir=${1:-$GOAL3_RUN_DIR}
    local snapshot_root="$out_dir/source-snapshot"
    local snapshot="$snapshot_root/goal-3"
    local manifest="$out_dir/source-snapshot.sha256"
    mkdir -p "$snapshot_root"
    rsync -a \
        --exclude logs \
        --exclude __pycache__ \
        --exclude '*.pyc' \
        "$GOAL3_REPO_ROOT/goal-3/" "$snapshot/"
    (
        cd "$snapshot"
        find . -type f | LC_ALL=C sort | while IFS= read -r path; do
            if command -v sha256sum >/dev/null 2>&1; then
                sha256sum "$path"
            else
                shasum -a 256 "$path"
            fi
        done
    ) >"$manifest"
}

goal3_ensure_runtime_requirements() {
    local out_dir=${1:-$GOAL3_RUN_DIR}
    mkdir -p "$out_dir/runtime-setup"
    python - <<'PY' >"$out_dir/runtime-setup/imports-before.json" 2>"$out_dir/runtime-setup/imports-before.err" || true
import importlib.util
import json

mods = ["torch", "triton", "sentencepiece", "brotli", "flash_attn_interface"]
print(json.dumps({name: importlib.util.find_spec(name) is not None for name in mods}, indent=2, sort_keys=True))
PY
    if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("flash_attn_interface") else 1)
PY
    then
        if [[ "${GOAL3_ALLOW_RUNTIME_FA3_INSTALL:-1}" != "1" ]]; then
            echo "flash_attn_interface missing and GOAL3_ALLOW_RUNTIME_FA3_INSTALL!=1" >&2
            return 2
        fi
        echo "flash_attn_interface missing; attempting runtime FA3 install from $GOAL3_FA3_FIND_LINKS" \
            | tee "$out_dir/runtime-setup/fa3-install.txt"
        python -m pip install --no-deps flash_attn_3 --find-links "$GOAL3_FA3_FIND_LINKS" \
            >>"$out_dir/runtime-setup/fa3-install.txt" 2>&1
    fi
    python - <<'PY' >"$out_dir/runtime-setup/imports-after.json"
import importlib.util
import json
import shutil

mods = ["torch", "triton", "sentencepiece", "brotli", "flash_attn_interface"]
payload = {name: importlib.util.find_spec(name) is not None for name in mods}
payload["lrzip"] = shutil.which("lrzip")
print(json.dumps(payload, indent=2, sort_keys=True))
missing = [name for name, ok in payload.items() if name != "lrzip" and not ok]
if missing or payload["lrzip"] is None:
    raise SystemExit(1)
PY
}

goal3_write_final_status() {
    local status=${1:-1}
    local reason=${2:-unknown}
    local out_dir=${3:-$GOAL3_RUN_DIR}
    mkdir -p "$out_dir"
    python - "$out_dir" "$status" "$reason" "$(goal3_timestamp)" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
exit_code = int(sys.argv[2])
reason = sys.argv[3]
timestamp = sys.argv[4]
path = out_dir / "final-status.json"
if path.exists():
    raise SystemExit(0)
payload = {
    "status": "passed" if exit_code == 0 else "failed",
    "exit_code": exit_code,
    "reason": reason,
    "timestamp": timestamp,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

goal3_maybe_run_codex_repair() {
    local failure_status=${1:-1}
    local reason=${2:-early_exit}
    local out_dir=${3:-$GOAL3_RUN_DIR}

    if [[ "${GOAL3_ENABLE_REPAIR_AGENT:-0}" != "1" ]]; then
        return 0
    fi

    mkdir -p "$out_dir"
    local codex_bin=${GOAL3_CODEX_BIN:-/nfs/stak/users/peterj29/.local/bin/codex}
    local status_file="$out_dir/codex-repair-status.txt"
    if [[ ! -x "$codex_bin" ]]; then
        {
            echo "status=skipped"
            echo "reason=codex_not_executable"
            echo "codex_bin=$codex_bin"
        } >"$status_file"
        return 0
    fi

    local repair_prompt="$out_dir/repair-prompt.md"
    {
        if [[ -f "$GOAL3_REPO_ROOT/goal-3/0-prompt.md" ]]; then
            cat "$GOAL3_REPO_ROOT/goal-3/0-prompt.md"
        else
            echo "# Goal 3 Repair"
        fi
        printf '\n\n## Repair-Agent Bounds\n\n'
        printf 'Slurm job: %s\n' "${SLURM_JOB_ID:-unknown}"
        printf 'Run directory: %s\n' "$out_dir"
        printf 'Failure status: %s\n' "$failure_status"
        printf 'Failure reason: %s\n\n' "$reason"
        printf '%s\n' '- You are running inside the failing Goal 3 H100 campaign allocation.'
        printf '%s\n' '- Do not submit new Slurm jobs.'
        printf '%s\n' '- Do not launch broad sweeps.'
        printf '%s\n' '- Only diagnose and patch the current Goal 3 run path.'
        printf '%s\n' '- Run at most one bounded smoke after a patch.'
        printf '%s\n' '- Write findings to goal-3/status.md and the current run directory.'
    } >"$repair_prompt"

    set +e
    timeout "${GOAL3_REPAIR_TIMEOUT:-20m}" "$codex_bin" exec \
        --cd "$GOAL3_REPO_ROOT" \
        "$(cat "$repair_prompt")" \
        >"$out_dir/codex-repair.stdout" \
        2>"$out_dir/codex-repair.stderr"
    local repair_status=$?
    set -e

    {
        echo "status=$repair_status"
        echo "timestamp=$(goal3_timestamp)"
        echo "codex_bin=$codex_bin"
        echo "timeout=${GOAL3_REPAIR_TIMEOUT:-20m}"
        echo "prompt=$repair_prompt"
        echo "stdout=$out_dir/codex-repair.stdout"
        echo "stderr=$out_dir/codex-repair.stderr"
    } >"$status_file"
    return "$repair_status"
}

goal3_cleanup_local_workspace() {
    local local_root=${GOAL3_LOCAL_ROOT:-}
    if [[ -z "$local_root" ]]; then
        return 0
    fi
    case "$local_root" in
        /scratch/"$USER"/"$SLURM_JOB_ID"/goal3)
            rm -rf -- "$local_root"
            ;;
        *)
            echo "refusing unsafe GOAL3_LOCAL_ROOT cleanup: $local_root" >&2
            return 1
            ;;
    esac
}

goal3_prepare_local_workspace() {
    local local_root=${GOAL3_LOCAL_ROOT:-/scratch/$USER/${SLURM_JOB_ID:-manual}/goal3}
    local manifest=${GOAL3_RUN_DIR:-$PWD}/scratch-stage.txt
    mkdir -p "$local_root" "$(dirname "$manifest")"
    export GOAL3_LOCAL_ROOT=$local_root

    {
        echo "timestamp=$(goal3_timestamp)"
        echo "local_root=$local_root"
        echo "source_stage_enabled=${GOAL3_STAGE_SOURCE_TO_SCRATCH:-1}"
        echo "data_stage_enabled=${GOAL3_STAGE_DATA_TO_SCRATCH:-1}"
    } >"$manifest"

    if [[ "${GOAL3_STAGE_SOURCE_TO_SCRATCH:-1}" == "1" ]]; then
        local local_stage="$local_root/stage/primary-qmlp"
        mkdir -p "$local_stage"
        rsync -a "$GOAL3_STAGE_DIR/" "$local_stage/"
        echo "GOAL3_STAGE_DIR=$local_stage" >>"$manifest"
        export GOAL3_STAGE_DIR=$local_stage
    fi

    if [[ "${GOAL3_STAGE_DATA_TO_SCRATCH:-1}" == "1" ]]; then
        local local_data_root="$local_root/data"
        local local_tok_root="$local_root/tokenizers"
        mkdir -p "$local_data_root/sp8192" "$local_data_root/sp16384" "$local_tok_root"

        rsync -a "$GOAL3_SP8192_DATA/" "$local_data_root/sp8192/"
        rsync -a "$GOAL3_SP16384_DATA/" "$local_data_root/sp16384/"
        cp "$GOAL3_SP8192_TOKENIZER" "$local_tok_root/sp8192.model"
        cp "$GOAL3_SP16384_TOKENIZER" "$local_tok_root/sp16384.model"

        export GOAL3_SP8192_DATA="$local_data_root/sp8192"
        export GOAL3_SP16384_DATA="$local_data_root/sp16384"
        export GOAL3_SP8192_TOKENIZER="$local_tok_root/sp8192.model"
        export GOAL3_SP16384_TOKENIZER="$local_tok_root/sp16384.model"

        {
            echo "GOAL3_SP8192_DATA=$GOAL3_SP8192_DATA"
            echo "GOAL3_SP16384_DATA=$GOAL3_SP16384_DATA"
            echo "GOAL3_SP8192_TOKENIZER=$GOAL3_SP8192_TOKENIZER"
            echo "GOAL3_SP16384_TOKENIZER=$GOAL3_SP16384_TOKENIZER"
        } >>"$manifest"
    fi
}

goal3_base_env() {
    export CASEOPS_ENABLED=1
    export NUM_LAYERS=11
    export XSA_LAST_N=11
    export MODEL_DIM=512
    export NUM_HEADS=8
    export NUM_KV_HEADS=4
    export MLP_MULT=4.0
    export ITERATIONS=${ITERATIONS:-20000}
    export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}
    export PHASED_TTT_PREFIX_DOCS=${PHASED_TTT_PREFIX_DOCS:-2500}
    export PHASED_TTT_NUM_PHASES=${PHASED_TTT_NUM_PHASES:-3}
    export EMBED_BITS=${EMBED_BITS:-7}
    export MATRIX_LR=${MATRIX_LR:-0.026}
    export MIN_LR=${MIN_LR:-0.1}
    export MLP_CLIP_SIGMAS=${MLP_CLIP_SIGMAS:-11.5}
    export ATTN_CLIP_SIGMAS=${ATTN_CLIP_SIGMAS:-13.0}
    export EMBED_CLIP_SIGMAS=${EMBED_CLIP_SIGMAS:-14.0}
    export GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-0.3}
    export TTT_CHUNK_SIZE=${TTT_CHUNK_SIZE:-48}
    export WARMUP_STEPS=${WARMUP_STEPS:-20}
    export MUON_BACKEND_STEPS=${MUON_BACKEND_STEPS:-5}
    export GLOBAL_TTT_MOMENTUM=${GLOBAL_TTT_MOMENTUM:-0.9}
    export WARMDOWN_FRAC=${WARMDOWN_FRAC:-0.85}
    export BETA2=${BETA2:-0.99}
    export TTT_BETA2=${TTT_BETA2:-0.99}
    export TTT_WEIGHT_DECAY=${TTT_WEIGHT_DECAY:-0.5}
    export TTT_LORA_RANK=${TTT_LORA_RANK:-80}
    export SPARSE_ATTN_GATE_SCALE=${SPARSE_ATTN_GATE_SCALE:-0.5}
    export GPTQ_RESERVE_SECONDS=${GPTQ_RESERVE_SECONDS:-8.0}
    export GPTQ_CALIBRATION_BATCHES=${GPTQ_CALIBRATION_BATCHES:-16}
    export VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0}
    export GATED_ATTN_QUANT_GATE=${GATED_ATTN_QUANT_GATE:-1}
    export SPARSE_ATTN_GATE_ENABLED=${SPARSE_ATTN_GATE_ENABLED:-1}
    export GATE_WINDOW=${GATE_WINDOW:-12}
    export SMEAR_GATE_ENABLED=${SMEAR_GATE_ENABLED:-1}
    export LQER_ENABLED=${LQER_ENABLED:-1}
    export LQER_ASYM_ENABLED=${LQER_ASYM_ENABLED:-1}
    export LQER_RANK=${LQER_RANK:-4}
    export LQER_FACTOR_BITS=${LQER_FACTOR_BITS:-4}
    export LQER_ASYM_GROUP=${LQER_ASYM_GROUP:-64}
    export LQER_TOP_K=${LQER_TOP_K:-3}
    export FUSED_CE_ENABLED=${FUSED_CE_ENABLED:-1}
    export COMPRESSOR=${COMPRESSOR:-pergroup}
    export NCCL_NET=${NCCL_NET:-Socket}
}

goal3_smoke_env() {
    export ITERATIONS=${ITERATIONS:-20}
    export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-120}
    export WARMUP_STEPS=${WARMUP_STEPS:-1}
    export VAL_DOC_FRACTION=${VAL_DOC_FRACTION:-0.01}
    export GPTQ_CALIBRATION_BATCHES=${GPTQ_CALIBRATION_BATCHES:-1}
    export TTT_ENABLED=${TTT_ENABLED:-0}
    export TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-1}
}

goal3_select_candidate() {
    local candidate=$1
    case "$candidate" in
        dense_sp8192|dense_sp8192_smoke)
            export VOCAB_SIZE=8192
            export DATA_PATH=$GOAL3_SP8192_DATA
            export TOKENIZER_PATH=$GOAL3_SP8192_TOKENIZER
            export QUAT_MLP=0
            export QUAT_MLP_IMPL=matrix
            ;;
        qmlp_sp8192|qmlp_sp8192_smoke)
            export VOCAB_SIZE=8192
            export DATA_PATH=$GOAL3_SP8192_DATA
            export TOKENIZER_PATH=$GOAL3_SP8192_TOKENIZER
            export QUAT_MLP=1
            export QUAT_MLP_IMPL=matrix
            ;;
        qmlp_sp16384|qmlp_sp16384_smoke|qmlp_sp16384_ttt_smoke)
            export VOCAB_SIZE=16384
            export DATA_PATH=$GOAL3_SP16384_DATA
            export TOKENIZER_PATH=$GOAL3_SP16384_TOKENIZER
            export QUAT_MLP=1
            export QUAT_MLP_IMPL=matrix
            ;;
        *)
            echo "unknown candidate: $candidate" >&2
            return 2
            ;;
    esac
    if [[ "$candidate" == *_smoke ]]; then
        goal3_smoke_env
    fi
    if [[ "$candidate" == *_ttt_smoke ]]; then
        export TTT_ENABLED=1
        export PHASED_TTT_NUM_PHASES=${PHASED_TTT_NUM_PHASES:-1}
        export PHASED_TTT_PREFIX_DOCS=${PHASED_TTT_PREFIX_DOCS:-16}
        export TTT_LORA_RANK=${TTT_LORA_RANK:-8}
        export TTT_BATCH_SIZE=${TTT_BATCH_SIZE:-8}
        export TTT_CHUNK_SIZE=${TTT_CHUNK_SIZE:-16}
        export TTT_EVAL_BATCHES=${TTT_EVAL_BATCHES:-1}
    fi
    goal3_base_env
}

goal3_assert_inputs() {
    test -d "$GOAL3_STAGE_DIR"
    test -f "$GOAL3_STAGE_DIR/train_gpt.py"
    test -d "$DATA_PATH"
    test -f "$TOKENIZER_PATH"
    test -f "$DATA_PATH/fineweb_val_000000.bin"
    test -f "$DATA_PATH/fineweb_val_bytes_000000.bin"
}
