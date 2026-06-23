#!/usr/bin/env bash

set -euo pipefail

GOAL3_REPO_ROOT=${GOAL3_REPO_ROOT:-/nfs/hpc/share/peterj29/pg/src/pg}
GOAL3_STAGE_DIR=${GOAL3_STAGE_DIR:-$GOAL3_REPO_ROOT/goal-3/stage/primary-qmlp}
GOAL3_RUN_ROOT=${GOAL3_RUN_ROOT:-/nfs/hpc/share/peterj29/pg/goal-3-runs}
GOAL3_ENV_DIR=${GOAL3_ENV_DIR:-/nfs/hpc/share/peterj29/pg/envs/goal3-cu128}
GOAL3_TOOLS_DIR=${GOAL3_TOOLS_DIR:-/nfs/hpc/share/peterj29/pg/tools}
GOAL3_ARTIFACT_LIMIT=${GOAL3_ARTIFACT_LIMIT:-16000000}

GOAL3_SP8192_DATA=${GOAL3_SP8192_DATA:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}
GOAL3_SP8192_TOKENIZER=${GOAL3_SP8192_TOKENIZER:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}
GOAL3_SP16384_DATA=${GOAL3_SP16384_DATA:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/datasets/datasets/fineweb10B_sp16384_lossless_caps_caseops_v1_reserved}
GOAL3_SP16384_TOKENIZER=${GOAL3_SP16384_TOKENIZER:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp16384/datasets/tokenizers/fineweb_16384_bpe_lossless_caps_caseops_v1_reserved.model}

export GOAL3_REPO_ROOT GOAL3_STAGE_DIR GOAL3_RUN_ROOT GOAL3_ENV_DIR
export GOAL3_TOOLS_DIR
export GOAL3_ARTIFACT_LIMIT
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

goal3_activate_env() {
    if [[ ! -d "$GOAL3_ENV_DIR" ]]; then
        echo "missing GOAL3_ENV_DIR=$GOAL3_ENV_DIR" >&2
        echo "Run goal-3/prepare-env.sbatch before H100 execution." >&2
        return 2
    fi
    # shellcheck disable=SC1091
    source "$GOAL3_ENV_DIR/bin/activate"
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
        qmlp_sp16384|qmlp_sp16384_smoke)
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
