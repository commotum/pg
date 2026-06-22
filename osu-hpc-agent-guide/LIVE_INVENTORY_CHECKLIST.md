# Live Inventory Checklist for OSU COE HPC

Run the companion collector on a submit node:

```bash
bash bin/osu-hpc-inventory.sh
```

It creates a timestamped directory under `inventory/`. The script is read-only and does not SSH to compute nodes.

## 1. Confirm the control plane

```bash
hostname -f
date --iso-8601=seconds
scontrol --version
sinfo
```

Record which submit hosts are online and whether Slurm is responding. Check the public status page separately:

https://it.engineering.oregonstate.edu/hpc/hpc-cluster-status-and-news

## 2. Confirm user associations

```bash
sacctmgr -nP show assoc where user="$USER" \
  format=Cluster,Account,User,Partition,DefaultQOS,QOS
```

Questions to resolve:

- Which accounts may this user charge?
- Is one account the default?
- Are partitions restricted on an association?
- Which QOS names apply?
- Are there group-level resource limits not visible in the public table?

If `sacctmgr` is not readable, inspect recent jobs:

```bash
sacct -u "$USER" -S now-90days -X \
  -o JobID,Account,Partition,QOS,State,Elapsed
```

## 3. Confirm partitions and limits

```bash
scontrol show partition -o
sinfo -o '%P|%a|%l|%D|%F|%G'
```

For each intended partition, capture:

```bash
scontrol show partition PARTITION
```

Check:

- `State=UP`
- maximum and default walltime
- allowed accounts/groups/QOS
- nodes assigned
- default memory behavior
- oversubscription policy
- preemption policy
- GRES and feature availability

The public guide's limits may be implemented through QOS rather than the partition object, so inspect associations/QOS too.

## 4. Reconcile node inventory

```bash
sinfo -N -a -o '%N|%P|%T|%c|%m|%G|%f'
scontrol show nodes -o
```

Resolve these known public-document uncertainties:

- active `dgx2-*` hosts and current count;
- whether `dgx2-3` remains removed;
- active `dgxh-*` hosts after June 2026 maintenance;
- exact identity and GRES of `cn-x-1`;
- `cn-v-*`, `cn-e*`, and `cn-a*` host-range/count mismatches;
- private GPU nodes omitted from the static inventory;
- drained/down nodes and their reasons.

Do not treat a drained or down node as available merely because it exists in `scontrol show nodes`.

## 5. Reconcile GPU types and MIG

```bash
sinfo -N -o '%N|%P|%T|%G|%f' \
  | grep -Ei 'gpu|h100|h200|l40s|a40|v100|rtx|t4|m60|vram'
```

For a small authorized allocation:

```bash
srun -p PARTITION --gres=gpu:1 --time=00:10:00 \
  --cpus-per-task=1 --mem=2G bash -lc '
    hostname
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
    nvidia-smi -L
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
      --format=csv
  '
```

Do not request one diagnostic allocation per node. One representative allocation per intended GPU class is normally enough.

Check:

- exact GRES spelling;
- generic versus typed GPU requests;
- MIG profile names;
- full-GPU versus MIG placement;
- visible memory;
- driver version;
- whether the desired partition is accessible with the chosen account.

## 6. Confirm OS and CPU features

```bash
sinfo -N -o '%N|%P|%T|%c|%m|%f'
```

For a representative allocation:

```bash
srun -p PARTITION --constraint=el9 --time=00:10:00 \
  --cpus-per-task=1 --mem=1G bash -lc '
    cat /etc/os-release
    lscpu
  '
```

Confirm feature names such as `el8`, `el9`, `avx2`, `avx512`, `epyc`, and generation labels before using them in scripts.

## 7. Confirm software modules

```bash
module -t avail 2>&1 | sort
module spider PACKAGE
```

For each project, record exact versions of:

- compiler;
- MPI;
- CUDA;
- Python or Conda;
- math libraries;
- licensed applications;
- build tools.

Do not update the guide's version list solely from a default module. Defaults can move; record all relevant versions and the chosen one.

## 8. Confirm CUDA and framework compatibility

Within a GPU allocation:

```bash
module purge
module load cuda/VERSION
nvidia-smi
nvcc --version

python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("runtime:", torch.version.cuda)
    print("available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print(repr(exc))
PY
```

Check the host driver, toolkit, framework runtime, custom extensions, and target GPU together.

## 9. Confirm storage and quotas

```bash
quota -s 2>/dev/null || true
disk-usage 2>/dev/null || true
df -hT "$HOME" "/nfs/hpc/share/$USER" 2>/dev/null || true
stat "$HOME" "/nfs/hpc/share/$USER" 2>/dev/null || true
```

Resolve the public 1 TB versus 1.5 TB `hpc-share` discrepancy through live quota reporting.

Inside a representative compute allocation:

```bash
df -hT /scratch
stat /scratch
hostname
```

Check:

- available local scratch;
- ownership and required subdirectory convention;
- whether `/nfs/hpc/HOSTNAME` exists;
- whether stage-out can reach `hpc-share`;
- actual project/archive destination.

Do not use `du` recursively as a routine monitor.

## 10. Validate a minimal CPU job

```bash
mkdir -p logs
job_id=$(sbatch --parsable templates/cpu.sbatch)
echo "$job_id"
squeue -j "$job_id"
```

After it exits:

```bash
sacct -j "$job_id" -X \
  -o JobID,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode
```

Confirm log creation, module loading, and stage-out.

## 11. Validate a minimal GPU job

Edit placeholders in `templates/gpu.sbatch`, then:

```bash
bash -n templates/gpu.sbatch
job_id=$(sbatch --parsable templates/gpu.sbatch)
echo "$job_id"
```

Confirm:

- the job reaches a compatible node;
- only requested devices are visible;
- the framework sees the GPU;
- no secrets are logged;
- output returns from local scratch.

## 12. Record reconciliation results

Update a project-local file rather than silently editing global assumptions:

```text
inventory date:
Slurm version:
accounts:
usable partitions:
partition/QOS ceilings:
active GPU classes:
GRES/feature spelling:
EL8/EL9 availability:
compiler/MPI stack:
CUDA/driver/framework matrix:
home quota:
hpc-share quota:
durable storage:
known maintenance/drains:
validated CPU job ID:
validated GPU job ID:
```

Retain the raw inventory report with the project run metadata. Re-run after maintenance, driver changes, Slurm changes, or a material module update.
