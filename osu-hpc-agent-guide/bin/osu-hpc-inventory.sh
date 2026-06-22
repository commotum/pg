#!/usr/bin/env bash
# Read-only inventory collector for the OSU College of Engineering HPC cluster.
# Run on a submit node. It does not SSH to compute nodes or submit jobs.

set -u
umask 077

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
short_host=$(hostname -s 2>/dev/null || hostname)
safe_user=${USER:-$(id -un)}
root=${1:-"inventory/${timestamp}-${short_host}-${safe_user}"}
mkdir -p "$root"

# Lmod is usually initialized by the login shell. Try standard initializers if not.
if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        if [[ -r "$init" ]]; then
            # shellcheck disable=SC1090
            source "$init"
            break
        fi
    done
fi

if ! command -v sinfo >/dev/null 2>&1 && type module >/dev/null 2>&1; then
    module load slurm >/dev/null 2>&1 || true
fi

section() {
    local filename=$1
    local title=$2
    shift 2
    (
        printf '# %s\n' "$title"
        printf '# captured: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
        printf '# command:'
        printf ' %q' "$@"
        printf '\n\n'
        "$@"
        rc=$?
        printf '\n# exit_status=%d\n' "$rc"
        exit "$rc"
    ) >"$root/$filename" 2>&1 || true
}

shell_section() {
    local filename=$1
    local title=$2
    local command=$3
    (
        printf '# %s\n' "$title"
        printf '# captured: %s\n' "$(date --iso-8601=seconds 2>/dev/null || date)"
        printf '# shell command: %s\n\n' "$command"
        bash -lc "$command"
        rc=$?
        printf '\n# exit_status=%d\n' "$rc"
        exit "$rc"
    ) >"$root/$filename" 2>&1 || true
}

{
    echo "# OSU HPC live inventory"
    echo
    echo "- Captured UTC: $timestamp"
    echo "- Host: $short_host"
    echo "- User: $safe_user"
    echo "- Output directory: $root"
    echo
    echo "This report is read-only. It contains no full environment dump and does"
    echo "not query compute hosts by SSH. Review files for local path or account"
    echo "information before sharing them outside the project."
} >"$root/README.md"

section "00-identity.txt" "Identity and kernel" bash -lc '
    date --iso-8601=seconds 2>/dev/null || date
    hostname -f 2>/dev/null || hostname
    whoami
    id
    uname -a
    printf "\nCurrent directory:\n"
    pwd
    printf "\nShell:\n"
    printf "%s\n" "${SHELL-unknown}"
'

section "01-os-release.txt" "Operating system on current host" bash -lc '
    for f in /etc/os-release /etc/redhat-release; do
        if [[ -r "$f" ]]; then
            echo "## $f"
            cat "$f"
        fi
    done
'

section "02-slurm-version.txt" "Slurm client and controller" bash -lc '
    command -v scontrol || true
    scontrol --version
    printf "\nController/config summary:\n"
    scontrol show config | grep -E "^(ClusterName|SlurmctldHost|SlurmctldPort|SlurmctldParameters|SelectType|SelectTypeParameters|PreemptMode|PreemptType|PriorityType|SchedulerType|SlurmctldTimeout|SlurmdTimeout|SlurmUser|StateSaveLocation)" || true
'

section "10-sinfo-summary.txt" "Partition summary" \
    sinfo -a -o '%20P|%10a|%12l|%8D|%24F|%30G'

section "11-sinfo-nodes.txt" "Node state, resources, GRES, and features" \
    sinfo -a -N -o '%30N|%22P|%12T|%8c|%12m|%45G|%120f'

section "12-partitions-oneline.txt" "Full live partition definitions" \
    scontrol show partition -o

section "13-nodes-oneline.txt" "Full live node definitions" \
    scontrol show nodes -o

section "14-reservations.txt" "Slurm reservations" \
    scontrol show reservations -o

section "15-frontends.txt" "Slurm front-end definitions, if used" \
    scontrol show frontends -o

section "20-user-associations.txt" "Visible Slurm associations for current user" bash -lc '
    sacctmgr -nP show assoc where user="$USER" \
      format=Cluster,Account,User,Partition,DefaultQOS,QOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmit 2>/dev/null \
    || sacctmgr -nP show assoc where user="$USER" \
      format=Cluster,Account,User,Partition,DefaultQOS,QOS 2>/dev/null \
    || true
'

section "21-qos.txt" "Visible Slurm QOS definitions" bash -lc '
    sacctmgr -nP show qos \
      format=Name,Priority,Preempt,PreemptMode,MaxWall,MaxJobsPU,MaxSubmitJobsPU,MaxTRESPU,GrpTRES,MaxTRESMinsPU 2>/dev/null \
    || sacctmgr -nP show qos format=Name,Priority,MaxWall 2>/dev/null \
    || true
'

section "22-fairshare.txt" "Visible fair-share information" bash -lc '
    sshare -U 2>/dev/null || sshare -u "$USER" 2>/dev/null || true
'

section "23-priority.txt" "Current-user job priorities" bash -lc '
    sprio -u "$USER" 2>/dev/null || true
'

section "24-current-jobs.txt" "Current jobs for user" \
    squeue -u "$safe_user" -o '%.18i|%.16P|%.30j|%.12a|%.2t|%.12M|%.12l|%.6D|%.20R'

section "25-recent-accounting.txt" "Recent job accounting" bash -lc '
    sacct -u "$USER" -S now-7days -X \
      -o JobID,JobName,Account,Partition,QOS,State,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS,ExitCode 2>/dev/null \
    || true
'

section "30-module-list.txt" "Modules currently loaded" bash -lc '
    type module 2>&1 || true
    module -t list 2>&1 || true
'

section "31-module-avail.txt" "Complete visible Lmod module inventory" bash -lc '
    module -t avail 2>&1 | sort
'

section "32-module-spider-selected.txt" "Selected module families" bash -lc '
    for package in cuda gcc llvm intel oneapi nvhpc openmpi mpich python conda miniforge apptainer matlab mathematica ansys starccm gurobi; do
        echo
        echo "## module spider $package"
        module spider "$package" 2>&1 || true
    done
'

section "33-samples.txt" "Site sample-job directory" bash -lc '
    if [[ -d /apps/samples ]]; then
        find /apps/samples -maxdepth 2 -type f -printf "%p\n" 2>/dev/null | sort
    else
        echo "/apps/samples is not visible on this host"
    fi
'

section "40-storage.txt" "Storage mounts and capacity" bash -lc '
    paths=("$HOME" "/nfs/hpc/share/$USER" "/scratch")
    for path in "${paths[@]}"; do
        echo
        echo "## $path"
        if [[ -e "$path" ]]; then
            stat -c "path=%n type=%F owner=%U group=%G mode=%A" "$path" 2>/dev/null || true
            df -hT "$path" 2>/dev/null || true
            findmnt -T "$path" 2>/dev/null || true
        else
            echo "not present"
        fi
    done
'

section "41-quota.txt" "Visible quota tools" bash -lc '
    echo "## quota -s"
    quota -s 2>&1 || true
    echo
    echo "## disk-usage"
    if command -v disk-usage >/dev/null 2>&1; then
        disk-usage 2>&1 || true
    else
        echo "disk-usage command not found"
    fi
'

section "42-hpc-share-link.txt" "HPC share path and home symlink" bash -lc '
    ls -ld "$HOME/hpc-share" "/nfs/hpc/share/$USER" 2>&1 || true
    readlink -f "$HOME/hpc-share" 2>/dev/null || true
'

section "50-submit-processes.txt" "Current user's processes on this host" \
    ps -u "$safe_user" -o pid,ppid,etime,%cpu,%mem,rss,vsz,stat,comm,args --sort=-%cpu

section "51-slurm-environment.txt" "SLURM variables in current shell" bash -lc '
    env | grep "^SLURM_" | sort || true
'

if [[ -n ${SLURM_JOB_ID-} ]] && command -v nvidia-smi >/dev/null 2>&1; then
    section "60-gpu-runtime.txt" "GPU details inside current allocation" bash -lc '
        echo "SLURM_JOB_ID=${SLURM_JOB_ID-}"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
        hostname -f 2>/dev/null || hostname
        nvidia-smi -L
        nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
        nvidia-smi
    '
else
    {
        echo "# GPU runtime inspection"
        echo
        echo "Skipped: this shell is not both inside a Slurm allocation and able to run nvidia-smi."
        echo "The script intentionally does not request a GPU or probe compute nodes."
    } >"$root/60-gpu-runtime.txt"
fi

# Concise machine-readable extracts useful for diffing successive inventories.
sinfo -a -N -h -o '%N|%P|%T|%c|%m|%G|%f' \
    >"$root/nodes.psv" 2>"$root/nodes.psv.err" || true
sinfo -a -h -o '%P|%a|%l|%D|%F|%G' \
    >"$root/partitions.psv" 2>"$root/partitions.psv.err" || true

{
    echo
    echo "Inventory complete: $root"
    echo "Review README.md and the numbered files."
} | tee "$root/COMPLETE.txt"
