#!/usr/bin/env bash
set -euo pipefail

SMOKE_ROOT=${SMOKE_ROOT:-/nfs/hpc/share/peterj29/pg/runs/goal2-phase3-smokes}
RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
HARNESS=${HARNESS:-$REPO_PATH/goal-2/2-a40-harness.sbatch}
VOCABS=${VOCABS:-"1024 2048 4096 8192 16384 32768"}
MODELS=${MODELS:-"dense qmlp"}
SEEDS=${SEEDS:-"42 0 1"}
BYTE_CAP=${BYTE_CAP:-16000000}
SUBMIT=${SUBMIT:-0}
ALLOW_OVER_BUDGET=${ALLOW_OVER_BUDGET:-0}
DIAGNOSTIC_VOCABS=${DIAGNOSTIC_VOCABS:-"32768"}
LEDGER=${LEDGER:-$RUN_ROOT/submitted.tsv}

mkdir -p "$RUN_ROOT"
touch "$LEDGER"

case "$SUBMIT" in
    0|1) ;;
    *) echo "SUBMIT must be 0 or 1" >&2; exit 2 ;;
esac
case "$ALLOW_OVER_BUDGET" in
    0|1) ;;
    *) echo "ALLOW_OVER_BUDGET must be 0 or 1" >&2; exit 2 ;;
esac

if [[ ! -f "$HARNESS" ]]; then
    echo "missing harness: $HARNESS" >&2
    exit 2
fi

caseops_root_for_vocab() {
    local vocab=$1
    if [[ "$vocab" == "8192" ]]; then
        printf '%s\n' "/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp8192-patched"
    else
        printf '%s\n' "/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp${vocab}"
    fi
}

metric_value() {
    local file=$1
    local key=$2
    awk -F= -v key="$key" '$1 == key {print substr($0, length(key) + 2)}' "$file" | tail -1
}

already_submitted() {
    local vocab=$1
    local model=$2
    local seed=$3
    awk -F '\t' -v vocab="$vocab" -v model="$model" -v seed="$seed" \
        '$1 == vocab && $2 == model && $3 == seed {found=1} END{exit found ? 0 : 1}' \
        "$LEDGER"
}

is_diagnostic_vocab() {
    local vocab=$1
    local diagnostic
    for diagnostic in $DIAGNOSTIC_VOCABS; do
        if [[ "$diagnostic" == "$vocab" ]]; then
            return 0
        fi
    done
    return 1
}

find_smoke_metrics() {
    local vocab=$1
    local model=$2
    local best=""
    while IFS= read -r metrics; do
        [[ -f "$metrics" ]] || continue
        mvocab=$(metric_value "$metrics" vocab_size)
        mmodel=$(metric_value "$metrics" model_variant)
        if [[ "$mvocab" == "$vocab" && "$mmodel" == "$model" ]]; then
            best="$metrics"
        fi
    done < <(find "$SMOKE_ROOT" -mindepth 2 -maxdepth 2 -name metrics.env 2>/dev/null | sort)
    printf '%s\n' "$best"
}

for vocab in $VOCABS; do
    for model in $MODELS; do
        metrics=$(find_smoke_metrics "$vocab" "$model")
        if [[ -z "$metrics" ]]; then
            echo "SKIP sp${vocab} ${model}: no Phase 3 metrics"
            continue
        fi

        run_dir=$(dirname "$metrics")
        bytes=$(metric_value "$metrics" total_submission_bytes)
        ttt_seen=$(metric_value "$metrics" ttt_seen)

        if [[ ! -s "$run_dir/COMPLETE.txt" ]]; then
            echo "SKIP sp${vocab} ${model}: smoke COMPLETE.txt missing"
            continue
        fi
        if [[ -z "$bytes" ]]; then
            echo "SKIP sp${vocab} ${model}: smoke total_submission_bytes=missing cap=$BYTE_CAP"
            continue
        fi
        diagnostic_vocab=0
        if is_diagnostic_vocab "$vocab"; then
            diagnostic_vocab=1
        fi

        if [[ "$bytes" -gt "$BYTE_CAP" && "$ALLOW_OVER_BUDGET" != "1" && "$diagnostic_vocab" != "1" ]]; then
            echo "SKIP sp${vocab} ${model}: smoke total_submission_bytes=${bytes:-missing} cap=$BYTE_CAP"
            continue
        fi
        if [[ "$bytes" -gt "$BYTE_CAP" ]]; then
            echo "DIAGNOSTIC sp${vocab} ${model}: over budget bytes=$bytes cap=$BYTE_CAP"
        fi
        if [[ "$ttt_seen" != "0" ]]; then
            echo "SKIP sp${vocab} ${model}: smoke ttt_seen=$ttt_seen"
            continue
        fi

        caseops_root=$(caseops_root_for_vocab "$vocab")
        for seed in $SEEDS; do
            if already_submitted "$vocab" "$model" "$seed"; then
                echo "SKIP sp${vocab} ${model} seed=${seed}: already in $LEDGER"
                continue
            fi
            export_arg="ALL,HARNESS_MODE=benchmark,MODEL_VARIANT=${model},VOCAB_SIZE=${vocab},SEED_VALUE=${seed},RUN_ROOT=${RUN_ROOT},CASEOPS_ROOT=${caseops_root}"
            echo "READY sp${vocab} ${model} seed=${seed}: smoke=$(basename "$run_dir") bytes=$bytes"
            if [[ "$SUBMIT" == "1" ]]; then
                job_id=$(sbatch --parsable \
                    --job-name="g2b-${model}-${vocab}-${seed}" \
                    --output="$RUN_ROOT/slurm-%j.out" \
                    --error="$RUN_ROOT/slurm-%j.err" \
                    --export="$export_arg" \
                    "$HARNESS")
                echo "SUBMITTED sp${vocab} ${model} seed=${seed}: $job_id"
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$vocab" "$model" "$seed" "$job_id" "$(date -Iseconds 2>/dev/null || date)" "$metrics" >>"$LEDGER"
            fi
        done
    done
done
