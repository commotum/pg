#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: run_candidate.sh CANDIDATE [SEED]" >&2
    exit 2
fi

candidate=$1
seed=${2:-42}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$script_dir/common.sh"

goal3_init_run_dir
goal3_activate_env
goal3_select_candidate "$candidate"

if [[ "$candidate" == *_smoke ]]; then
    export ITERATIONS=${GOAL3_SMOKE_ITERATIONS:-20}
    export MAX_WALLCLOCK_SECONDS=${GOAL3_SMOKE_MAX_WALLCLOCK_SECONDS:-120}
    export WARMUP_STEPS=${GOAL3_SMOKE_WARMUP_STEPS:-1}
    export VAL_DOC_FRACTION=${GOAL3_SMOKE_VAL_DOC_FRACTION:-0.01}
    export GPTQ_CALIBRATION_BATCHES=${GOAL3_SMOKE_GPTQ_CALIBRATION_BATCHES:-1}
    export TTT_ENABLED=${GOAL3_SMOKE_TTT_ENABLED:-0}
    export TRAIN_LOG_EVERY=${GOAL3_SMOKE_TRAIN_LOG_EVERY:-1}
    candidate_timeout=${GOAL3_SMOKE_TIMEOUT:-15m}
else
    candidate_timeout=${GOAL3_FULL_TIMEOUT:-35m}
fi

export SEED=$seed
export RUN_ID=${RUN_ID:-${candidate}_seed${seed}}
candidate_dir="$GOAL3_RUN_DIR/candidates/$candidate/seed_$seed"
mkdir -p "$candidate_dir"
export ARTIFACT_DIR=$candidate_dir

goal3_assert_inputs

{
    echo "timestamp=$(goal3_timestamp)"
    echo "candidate=$candidate"
    echo "seed=$seed"
    echo "run_id=$RUN_ID"
    echo "stage_dir=$GOAL3_STAGE_DIR"
    echo "artifact_dir=$ARTIFACT_DIR"
    echo "timeout=$candidate_timeout"
    echo "VOCAB_SIZE=$VOCAB_SIZE"
    echo "DATA_PATH=$DATA_PATH"
    echo "TOKENIZER_PATH=$TOKENIZER_PATH"
    echo "QUAT_MLP=$QUAT_MLP"
    echo "QUAT_MLP_IMPL=$QUAT_MLP_IMPL"
    echo "ITERATIONS=$ITERATIONS"
    echo "MAX_WALLCLOCK_SECONDS=$MAX_WALLCLOCK_SECONDS"
    echo "GPTQ_RESERVE_SECONDS=$GPTQ_RESERVE_SECONDS"
    echo "GPTQ_CALIBRATION_BATCHES=$GPTQ_CALIBRATION_BATCHES"
    echo "TTT_ENABLED=${TTT_ENABLED:-1}"
    echo "VAL_DOC_FRACTION=${VAL_DOC_FRACTION:-1.0}"
} >"$candidate_dir/env.txt"

cd "$GOAL3_STAGE_DIR"

set +e
timeout "$candidate_timeout" \
    torchrun --standalone --nproc_per_node="${GOAL3_NPROC_PER_NODE:-8}" train_gpt.py \
    > >(tee "$candidate_dir/stdout.log") \
    2> >(tee "$candidate_dir/stderr.log" >&2)
exit_code=$?
set -e

python "$script_dir/parse_train_log.py" \
    "$candidate_dir/stdout.log" \
    "$candidate_dir/summary.json" || true

python - "$candidate_dir/summary.json" "$GOAL3_ARTIFACT_LIMIT" "$exit_code" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
limit = int(sys.argv[2])
exit_code = int(sys.argv[3])
payload = json.loads(summary.read_text()) if summary.exists() else {}
total = payload.get("total_submission_bytes")
under = total is not None and total < limit
status = {
    "exit_code": exit_code,
    "total_submission_bytes": total,
    "artifact_limit": limit,
    "artifact_under_limit": under if total is not None else None,
}
Path(summary.parent / "status.json").write_text(json.dumps(status, indent=2) + "\n")
if exit_code != 0:
    raise SystemExit(exit_code)
if total is not None and not under:
    raise SystemExit(80)
PY
