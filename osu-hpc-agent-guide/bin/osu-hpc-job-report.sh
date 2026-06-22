#!/usr/bin/env bash
# Summarize a Slurm job without changing it.

set -u

if [[ $# -ne 1 ]] || [[ ! $1 =~ ^[0-9]+([_.][0-9]+)?$ ]]; then
    echo "usage: $0 JOBID" >&2
    exit 2
fi

job_id=$1

heading() {
    printf '\n===== %s =====\n' "$1"
}

heading "timestamp"
date --iso-8601=seconds 2>/dev/null || date
hostname -f 2>/dev/null || hostname

heading "squeue"
squeue -j "$job_id" \
  -o '%.18i|%.16P|%.30j|%.12a|%.2t|%.12M|%.12l|%.6D|%.30R' 2>&1 || true

heading "scontrol show job"
scontrol show job "$job_id" 2>&1 || true

heading "sacct"
sacct -j "$job_id" -X \
  -o JobID,JobName,Account,Partition,QOS,State,Reason,Elapsed,Timelimit,AllocCPUS,ReqCPUS,ReqMem,MaxRSS,MaxVMSize,AllocTRES,ReqTRES,ExitCode,NodeList \
  2>&1 || \
sacct -j "$job_id" -X \
  -o JobID,JobName,Account,Partition,State,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS,ExitCode,NodeList \
  2>&1 || true

heading "seff"
if command -v seff >/dev/null 2>&1; then
    seff "$job_id" 2>&1 || true
else
    echo "seff is not available"
fi

heading "live sstat"
sstat -j "${job_id}.batch" \
  --format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize,AveDiskRead,AveDiskWrite \
  2>&1 || true

heading "interpretation reminders"
cat <<'EOF'
- A job leaving squeue is not proof of success; inspect sacct State and ExitCode.
- COMPLETED with expected validated outputs is success.
- OUT_OF_MEMORY: inspect MaxRSS and application memory behavior.
- TIMEOUT: checkpoint, split, optimize, or request a permitted longer time.
- PREEMPTED/CANCELLED on preempt: resume from a validated checkpoint.
- NODE_FAIL: preserve logs and retry from a checkpoint; report persistent hardware symptoms.
- MaxRSS can be step-specific and may underrepresent a multi-step or distributed job.
EOF
