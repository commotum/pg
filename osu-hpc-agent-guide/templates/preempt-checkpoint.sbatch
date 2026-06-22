#!/usr/bin/env bash
#SBATCH --job-name=osu-preempt-demo
#SBATCH --partition=preempt
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --signal=B:USR1@120
#SBATCH --output=osu-preempt-%j.out
#SBATCH --error=osu-preempt-%j.err

# Demonstrates checkpoint handling on the preempt partition.
# Replace the demo loop with an application that can resume from checkpoints.
# Do not assume requeue is enabled; this script preserves state and exits.

set -euo pipefail
umask 077

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm
fi

submit_dir=${SLURM_SUBMIT_DIR:-$PWD}
if [[ -d /nfs/hpc/share/$USER ]]; then
    shared_root=${RUN_ROOT:-/nfs/hpc/share/$USER/agent-runs}
else
    shared_root=${RUN_ROOT:-$submit_dir/agent-runs}
fi
run_dir="$shared_root/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
checkpoint_dir="$run_dir/checkpoints"
mkdir -p "$checkpoint_dir"

child_pid=
checkpoint_requested=0

write_checkpoint() {
    local reason=$1
    local tmp="$checkpoint_dir/state.tmp"
    {
        echo "job_id=$SLURM_JOB_ID"
        echo "reason=$reason"
        echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
        echo "last_step=$(cat "$run_dir/last-step.txt" 2>/dev/null || echo 0)"
    } >"$tmp"
    mv -f -- "$tmp" "$checkpoint_dir/state.txt"
    sync "$checkpoint_dir/state.txt" 2>/dev/null || true
}

on_usr1() {
    checkpoint_requested=1
    write_checkpoint "USR1-before-time-limit"
    [[ -z ${child_pid:-} ]] || kill -USR1 "$child_pid" 2>/dev/null || true
}
on_term() {
    checkpoint_requested=1
    write_checkpoint "TERM-or-preemption"
    [[ -z ${child_pid:-} ]] || kill -TERM "$child_pid" 2>/dev/null || true
}
trap on_usr1 USR1
trap on_term TERM INT

start_step=1
if [[ -s "$checkpoint_dir/state.txt" ]]; then
    last=$(awk -F= '$1=="last_step"{print $2}' "$checkpoint_dir/state.txt" | tail -1)
    if [[ $last =~ ^[0-9]+$ ]]; then
        start_step=$((last + 1))
    fi
fi

# Replace this demo with a real application invocation. The demo creates one
# atomic progress marker per minute and can resume from the last marker.
(
    set -euo pipefail
    for ((step=start_step; step<=60; step++)); do
        printf '%s\n' "$step" >"$run_dir/last-step.tmp"
        mv -f -- "$run_dir/last-step.tmp" "$run_dir/last-step.txt"
        sleep 60
    done
) &
child_pid=$!

status=0
wait "$child_pid" || status=$?

if [[ $checkpoint_requested -eq 1 ]]; then
    write_checkpoint "controlled-exit"
fi

exit "$status"
