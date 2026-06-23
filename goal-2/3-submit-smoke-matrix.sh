#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/goal2-phase3-smokes}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
HARNESS=${HARNESS:-$REPO_PATH/goal-2/2-a40-harness.sbatch}
SEED_VALUE=${SEED_VALUE:-42}
VOCABS=${VOCABS:-"1024 2048 4096 8192 16384"}
MODELS=${MODELS:-"dense qmlp"}
MIN_TRAIN_SHARDS=${MIN_TRAIN_SHARDS:-80}
SUBMIT=${SUBMIT:-0}

mkdir -p "$RUN_ROOT"

if [[ ! -f "$HARNESS" ]]; then
    echo "missing harness: $HARNESS" >&2
    exit 2
fi

case "$SUBMIT" in
    0|1) ;;
    *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;;
esac

caseops_root_for_vocab() {
    local vocab=$1
    if [[ "$vocab" == "8192" ]]; then
        printf '%s\n' "/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched"
    else
        printf '%s\n' "/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp${vocab}"
    fi
}

count_files() {
    local root=$1
    local pattern=$2
    find "$root" -type f -name "$pattern" 2>/dev/null | wc -l | tr -d ' '
}

for vocab in $VOCABS; do
    caseops_root=$(caseops_root_for_vocab "$vocab")
    caseops_out="$caseops_root/datasets"
    data_path="$caseops_out/datasets/fineweb10B_sp${vocab}_lossless_caps_caseops_v1_reserved"
    tokenizer_path="$caseops_out/tokenizers/fineweb_${vocab}_bpe_lossless_caps_caseops_v1_reserved.model"

    train_count=$(count_files "$data_path" 'fineweb_train_*.bin')
    val_count=$(find "$data_path" -type f -name 'fineweb_val_*.bin' ! -name 'fineweb_val_bytes_*.bin' 2>/dev/null | wc -l | tr -d ' ')
    byte_count=$(count_files "$data_path" 'fineweb_val_bytes_*.bin')

    if [[ ! -f "$tokenizer_path" || "$train_count" -lt "$MIN_TRAIN_SHARDS" || "$val_count" -lt 1 || "$byte_count" -lt 1 ]]; then
        echo "SKIP sp${vocab}: tokenizer=$([[ -f "$tokenizer_path" ]] && echo yes || echo no) train=$train_count val=$val_count bytes=$byte_count min_train=$MIN_TRAIN_SHARDS"
        continue
    fi

    for model in $MODELS; do
        export_arg="ALL,HARNESS_MODE=smoke,MODEL_VARIANT=${model},VOCAB_SIZE=${vocab},SEED_VALUE=${SEED_VALUE},RUN_ROOT=${RUN_ROOT},CASEOPS_ROOT=${caseops_root}"
        echo "READY sp${vocab} ${model}: $data_path"
        if [[ "$SUBMIT" == "1" ]]; then
            job_id=$(sbatch --parsable \
                --job-name="g2s-${model}-${vocab}" \
                --output="$RUN_ROOT/slurm-%j.out" \
                --error="$RUN_ROOT/slurm-%j.err" \
                --export="$export_arg" \
                "$HARNESS")
            echo "SUBMITTED sp${vocab} ${model}: $job_id"
        fi
    done
done
