# Agent's Guide to OSU College of Engineering HPC

**Edition:** 2026-06-20  
**Scope:** Oregon State University College of Engineering High Performance Computing Cluster  
**Audience:** OSU users, research software engineers, and coding agents such as Codex  
**Status:** Public-document baseline plus a live-verification procedure; not an administrator-issued policy

---

## 1. Purpose and operating model

The College of Engineering HPC cluster is a shared, heterogeneous research system managed with Slurm. It is not a single computer and it should not be treated like a conventional always-on development server. A user logs in to one of three submit nodes, prepares code and job descriptions there, and asks Slurm to allocate CPU, memory, GPU, local storage, and time on compute nodes.

A coding agent can operate effectively on this cluster only when it follows four principles:

1. **The submit nodes are control planes, not compute resources.** Editing, Git operations, modest file manipulation, module inspection, and job submission belong there. Compiling large projects, preprocessing large datasets, tests that consume material CPU or memory, model execution, and benchmarks belong inside a Slurm allocation.
2. **Resources must be explicit.** A GPU feature constraint does not allocate a GPU. A job must request one with `--gres=gpu:N` or the supported equivalent. Memory, CPU count, walltime, node count, partition, account, and relevant node features should also be explicit.
3. **The live cluster is authoritative.** The public pages contain several count, quota, and inventory inconsistencies and the cluster changes regularly. Query Slurm, Lmod, filesystems, and the status page immediately before using them.
4. **Every run should be reproducible and recoverable.** Pin modules or an environment, record the Git commit and system details, write outputs to a job-specific directory, copy valuable results off local scratch, and checkpoint work that can be preempted or exceed a maintenance window.

This document labels information as follows:

- **Documented:** stated by OSU's public HPC pages as reviewed on 2026-06-20.
- **Live-check required:** likely to change, internally inconsistent, or controlled by the user's account.
- **Agent practice:** a conservative operating rule intended to prevent wasted resources, data loss, or policy violations.

---

## 2. Immediate state check

Before any substantial work, run:

```bash
date
hostname -f
module list
sinfo
squeue -u "$USER"
sacctmgr -nP show assoc where user="$USER" \
  format=Cluster,Account,User,Partition,DefaultQOS,QOS 2>/dev/null || true
```

Then run the supplied inventory collector:

```bash
bash bin/osu-hpc-inventory.sh
```

### Public status at this edition's cutoff

OSU scheduled cluster-wide maintenance beginning June 15, 2026. The public status page reported on June 18 that the storage and InfiniBand upgrades and a Slurm update had completed, `submit-a` and `submit-b` were available, and `hpc-share` was online. Its June 18 6 p.m. update said that most basic EL9 nodes were up but EL9 GPU nodes, including DGX nodes, were still being restored due to driver conflicts.

That is a historical snapshot, not a claim about current availability. Check:

```bash
sinfo -o '%20P %10a %10l %6D %20F'
sinfo -N -o '%24N %18P %8T %8c %12m %32G %80f'
```

Do not submit a production run based solely on the hardware tables below.

---

## 3. Access, eligibility, and authentication

### 3.1 Who may use the cluster

The cluster's goal is to serve College of Engineering faculty and students. COE faculty, staff, and students can enable their Engineering account and activate HPC access in TEACH. OSU users outside COE may request access; students outside Engineering require faculty sponsorship. The current HPC home page also states that new users must attend the Intro to HPC training session.

Public-access workflow:

1. Create or enable an Engineering account.
2. In TEACH, select **High Performance Computing** under account tools.
3. Allow until the next business day for provisioning.
4. Non-COE users submit the access form with ONID, department, and advisor/sponsor.
5. For support, open an OSU support ticket and put **COE HPC** in the subject.

### 3.2 Network path

Off campus, connect to the OSU VPN or first SSH to the Engineering gateway:

```bash
ssh ONID@access.engr.oregonstate.edu
```

Then connect to a submit host:

```bash
ssh ONID@submit-a.hpc.engr.oregonstate.edu
ssh ONID@submit-b.hpc.engr.oregonstate.edu
ssh ONID@submit-c.hpc.engr.oregonstate.edu
```

The round-robin alias is also documented for file transfer:

```bash
ssh ONID@submit.hpc.engr.oregonstate.edu
```

Use SSH keys only if they comply with OSU requirements. Keep private keys on the client, use restrictive permissions, and never commit keys, tokens, or cookies to a repository.

### 3.3 Submit versus compute hosts

Direct SSH to a compute host is normally rejected by `pam_slurm_adopt` unless the user holds a current Slurm allocation on that host. The supported pattern is:

```bash
# On a submit node:
srun --time=01:00:00 --cpus-per-task=2 --mem=4G --pty bash

# Now inside the allocated compute shell:
hostname
```

Once allocated, direct SSH to that assigned node may be possible, but an agent should normally keep execution under `srun` so Slurm can account for and contain it.

### 3.4 Open OnDemand

The browser portal requires campus network or OSU VPN. OSU's pages currently expose two hostnames, apparently aliases:

- `https://ondemand.hpc.engr.oregonstate.edu`
- `https://submit.hpc.engr.oregonstate.edu`

Documented interactive applications include Basic Desktop, Advanced Desktop, Jupyter Notebook/Lab, MATLAB, Mathematica, RStudio, Ansys, and StarCCM+.

Use OnDemand for interactive GUI work. An unattended coding agent should favor `sbatch`, because batch jobs survive a VPN drop or submit-session termination.

---

## 4. High-level architecture

OSU describes a heterogeneous cluster with approximately:

- 130 compute servers;
- more than 5,500 CPU cores;
- more than 250 GPUs;
- more than 50 TB of aggregate RAM;
- roughly 500 TB of high-speed global scratch on a DDN AI400x2 Exascaler system;
- a private high-speed network in addition to the Engineering network;
- EDR and HDR InfiniBand on many newer systems;
- approximately 2,000 peak double-precision TFLOPS.

The cluster includes:

- three submit nodes;
- four DGX H-class systems: three H100 systems and one H200 system;
- a DGX-2 pool with V100 GPUs;
- many Dell and Exxact CPU/GPU nodes across Intel and AMD generations;
- local `/scratch` on compute nodes;
- global `hpc-share` visible through the cluster;
- Slurm for scheduling and resource accounting;
- Rocky Linux 8 and Rocky Linux 9 nodes;
- Lmod for software environments.

The system is deliberately heterogeneous. Binaries compiled with AVX-512 on a recent node may not run on an older node. A CUDA build suitable for H100 may not be compatible with a V100. A job must either target a known feature set or build conservatively.

---

## 5. Public hardware inventory

### 5.1 How to read this section

This is a transcription and normalization of OSU's public inventory, not a guarantee that every node is online or assigned to a partition accessible to the reader. Several public rows have mismatched host ranges and server counts. The current status page also reports failed or retired nodes that may still appear in static tables.

Use this live command before scheduling:

```bash
sinfo -N -a -o '%N|%P|%T|%c|%m|%G|%f'
```

For one candidate node:

```bash
scontrol show node NODE
```

### 5.2 Submit nodes

| Hosts | Model | CPU | RAM | Accelerator | Network |
|---|---|---:|---:|---|---|
| `submit-a`, `submit-b`, `submit-c` | Dell PowerEdge R740 | 2 × 16-core Intel Xeon Gold 6130, 2.10 GHz | 256 GB | None | Engineering Ethernet plus Mellanox EDR InfiniBand |

These hosts are for login, editing, source control, transfers, inspection, and Slurm control commands. They are not for sustained computation.

### 5.3 DGX H-class and DGX-2 systems

| Hosts | CPU | RAM | GPUs | Local NVMe | Network | Notes |
|---|---:|---:|---|---:|---|---|
| `dgxh-1`–`dgxh-3` | 2 × 56-core Xeon Platinum 8480CL | 2 TB | 8 × NVIDIA H100, 80 GB each | about 24 TB | 100 GbE + HDR InfiniBand | EL9/Sapphire Rapids class |
| `dgxh-4` | 2 × 56-core Xeon Platinum 8480CL | 2 TB | 8 × NVIDIA H200, 140 GB each | about 24 TB | 100 GbE + HDR InfiniBand | EL9/Sapphire Rapids class |
| `dgx2-*` | 2 × 24-core Xeon Platinum 8168 | 1.5 TB | 16 × NVIDIA V100, 32 GB each | about 28 TB | 100 GbE + EDR InfiniBand | Static page says five systems; summary says four; status says `dgx2-3` died in March 2026 |

**Live-check required:** Determine the active DGX-2 host list, current MIG configuration, and usable GRES names from Slurm. Do not infer them from the static count.

### 5.4 Recent Sapphire Rapids and AMD EPYC systems

| Hosts | CPU | RAM | GPUs | Local scratch | Fabric |
|---|---:|---:|---|---:|---|
| `cn-w-1` | 2 × 20-core Xeon Platinum 8462Y+ | 512 GB | 2 × H100 80 GB | about 3.5 TB NVMe | ConnectX-6 InfiniBand |
| `cn-w-2` | 2 × 20-core Xeon Platinum 8462Y+ | 512 GB | 4 × L40S 48 GB | about 3.5 TB NVMe | ConnectX-6 InfiniBand |
| `cn-x-1` | 2 × 128-core AMD EPYC 9755 | 512 GB | Public page says 8 × H100 with 140 GB each | about 3.5 TB NVMe | ConnectX-7 NDR InfiniBand |
| `cn-gpu10`–`cn-gpu12` | 2 × 64-core EPYC 7763 | 1 TB | 8 × L40S 48 GB per node | about 894 GB | ConnectX-6 InfiniBand |
| `cn-v-1`–`cn-v-9` | 2 × 48-core EPYC 9474F | 384 GB | None | about 400 GB | High-speed cluster fabric |
| `cn-u-1`–`cn-u-2` | 2 × 32-core EPYC 7543 | 512 GB | None | about 400 GB | High-speed cluster fabric |
| `cn-t-1` | 2 × 24-core EPYC 7313 | 256 GB | 3 × A40 48 GB | about 844 GB | High-speed cluster fabric |
| `cn-s-1`–`cn-s-5` | 2 × 24-core EPYC 74F3 | 256 GB | 2 × A40 48 GB per node | about 844 GB | High-speed cluster fabric |
| `cn-r-5`–`cn-r-6` | 2 × 32-core EPYC 7513 | 256 GB | 2 × A40 48 GB per node | about 725 GB | High-speed cluster fabric |
| `cn-r-1`–`cn-r-4` | 2 × 24-core EPYC 7F72 | 256 GB | 2 × A40 48 GB per node | about 725 GB | High-speed cluster fabric |

The `cn-x-1` row should be treated literally as a public-document anomaly: H100 is normally associated with other memory capacities, while H200 is associated with 141 GB-class memory. The public site says “H100” and “140 GB.” Query `scontrol show node cn-x-1` and `nvidia-smi -L` inside an allocation rather than silently correcting it.

The `cn-v-1`–`cn-v-9` host range implies nine hosts while the model/count text says eight. Live Slurm output resolves the discrepancy.

### 5.5 Intel Skylake and Cascade Lake systems

| Hosts | CPU | RAM | GPUs | Local scratch |
|---|---:|---:|---|---:|
| `sail-gpu0` | 2 × 24-core Xeon Gold 6248R | 768 GB | 8 × A40 48 GB | about 7 TB SSD |
| `soundwave` | 2 × 10-core Xeon Silver 4210R | 384 GB | None | about 5.3 TB SSD |
| `optimus` | 2 × 20-core Xeon Gold 6230 | 768 GB | 8 × RTX 6000; public page lists 22 GB each | about 345 GB SSD |
| `cn-gpu6`–`cn-gpu7` | 2 × 24-core Xeon Gold 6248R | 768 GB | 8 × RTX 8000; public page lists 44 GB each | about 7 TB SSD |
| `cn-gpu5` | 2 × 20-core Xeon Gold 6248 | 768 GB | 8 × RTX 8000; public page lists 44 GB each | about 7 TB SSD |
| `cn-q-1`–`cn-q-2` | 2 × 24-core Xeon Gold 6248R | 192 GB | None | about 345 GB |
| `cn-p-1` | 2 × 24-core Xeon Gold 6254R | 768 GB | 1 × V100S 32 GB | about 345 GB |
| `cn-o-1` | 2 × 24-core Xeon Gold 6254R | 192 GB | None | about 345 GB |
| `cn-n-1`–`cn-n-6` | 2 × 18-core Xeon Gold 6254 | 192 GB | None | about 345 GB |
| `cn-m-1` | 2 × 4-core Xeon Gold 5222 | 192 GB | 6 × Tesla T4; public page lists 15 GB each | about 155 GB SSD |
| `cn-m-2` | 2 × 8-core Xeon Silver 4215 | 192 GB | 2 × RTX 6000 24 GB | about 165 GB SSD |
| `cn-e41`, `cn-e43`, `cn-e44` | 2 × 22-core Xeon Gold 6152 | 128–768 GB | None | about 346 GB |
| `cn-e31`–`cn-e34` | 2 × 16-core Xeon Gold 6130 | 512 GB | None | about 346 GB |
| `cn-e21`–`cn-e24` | 2 × 14-core Xeon Gold 5120 | 256 GB | None | about 259 GB |
| `cn-e14`–`cn-e15` | 2 × 12-core Xeon Gold 6126 | 384 GB | None | about 259 GB |
| `cn-e11`–`cn-e13` | 2 × 12-core Xeon Silver 4116 | 256 GB | None | about 282 GB |
| `cn-e01`–`cn-e10` | 2 × 10-core Xeon Silver 4114 | 128 GB | None | about 203 GB |

Some public model-count labels for the `cn-e*` ranges do not match the ranges. Treat hostnames visible in `sinfo` as authoritative.

### 5.6 Haswell and Broadwell systems

| Hosts | CPU | RAM | GPUs | Local scratch |
|---|---:|---:|---|---:|
| `cn-d31` | 2 × 18-core Xeon E5-2695 v4 | 768 GB | None | about 203 GB |
| `cn-d21` | 2 × 8-core Xeon E5-2667 v4 | 512 GB | None | about 279 GB |
| `cn-d11`–`cn-d13` | 2 × 20-core Xeon E7-8870 v4 | 512 GB–1.5 TB | None | about 203 GB |
| `cn-d01` | 2 × 14-core Xeon E7-4830 v4 | 384 GB | None | about 203 GB |
| `cn-c31`–`cn-c33` | 2 × 12-core Xeon E5-2650L v3 | 512 GB | None | about 931 GB |
| `cn-c30` | 2 × 12-core Xeon E5-2670 v3 | 256 GB | 2 × Tesla M60 8 GB | about 3.8 TB |
| `cn-c20`–`cn-c24` | 2 × 10-core Xeon E5-2660 v3 | 128 GB | None | about 203 GB |
| `cn-c10`–`cn-c17` | 2 × 10-core Xeon E5-2660 v3 | 64–128 GB | None | about 193 GB |

### 5.7 Sandy Bridge and Ivy Bridge systems

| Hosts | CPU | RAM | GPUs | Local scratch |
|---|---:|---:|---|---:|
| `cn-b01`–`cn-b12` | 2 × 8-core Xeon E5-2650 v2 | 64 GB | None | about 338 GB |
| `cn-a20`–`cn-a26` | 2 × 8-core Xeon E5-2690 | 256 GB | None | about 203 GB |
| `cn-a11`–`cn-a15` | 2 × 8-core Xeon E5-2670 | 128 GB | None | about 203 GB |

Again, one public model-count label for the `cn-a11`–`cn-a15` range does not match the range.

---

## 6. GPU inventory and selection

### 6.1 Documented GPU families

The public hardware and partition pages identify access to these GPU families:

- NVIDIA H200
- NVIDIA H100
- NVIDIA L40S
- NVIDIA A40
- NVIDIA V100
- NVIDIA V100S
- NVIDIA RTX 8000
- NVIDIA RTX 6000
- NVIDIA Tesla T4
- NVIDIA Tesla M60
- RTX 2080 on at least one restricted/private partition
- MIG instances on some DGX H systems

The public aggregate says the cluster has more than 250 GPUs. The detailed static host list does not reconcile cleanly to that aggregate because it contains count anomalies, retired hosts, and likely private or omitted systems. Never derive a guaranteed available total from the static table.

### 6.2 Selection by Slurm feature

Documented node features include:

```text
m60 t4 rtx6000 rtx8000 v100 a40 h100 h200 l40s
vram40g vram80g vram140g
el8 el9 avx avx2 avx512 haswell broadwell skylake cascadelake epyc
```

A feature filters candidate nodes; it does not reserve an accelerator:

```bash
# Correct: one A40-class GPU
srun -p preempt --constraint=a40 --gres=gpu:1 \
  --time=01:00:00 --mem=32G --cpus-per-task=4 --pty bash

# Wrong: may select an A40 host but allocates zero GPUs
srun -p preempt --constraint=a40 --pty bash
```

Use `sinfo` to discover exact feature and GRES spelling:

```bash
sinfo -N -o '%N|%P|%T|%G|%f' | column -t -s '|'
```

### 6.3 MIG

OSU has used NVIDIA Multi-Instance GPU on some DGX H systems. Historical status notes mention 2g.20gb, 3g.40gb, 7g.80gb, and 7g.140gb profiles, followed by a reduction in MIG use because it disables some full-device capabilities. The current public partition table advertises `vram40g`, `vram80g`, and `vram140g` features.

MIG configuration is operationally mutable. Before writing a job around it:

```bash
sinfo -p dgxh,dgxh-ceoas -N -o '%N|%T|%G|%f'
scontrol show partition dgxh
```

Inside an allocation:

```bash
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
```

Do not hard-code a MIG GRES type until the live `Gres=` field confirms it.

### 6.4 GPU runtime inspection

Run these only on an allocated GPU node:

```bash
hostname
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-}"
which nvcc || true
nvcc --version || true
```

`nvidia-smi` reports the installed driver and visible hardware. `nvcc` reports the loaded CUDA toolkit. They are not the same thing.

### 6.5 CUDA compatibility

The public software list includes CUDA 11.x through 13.x. A March 2026 status entry says CUDA 13.1 was installed on most GPU nodes, but V100 support ended at CUDA 13.0, so `cuda/13.0` was set as the default module.

Agent rule:

- inspect the target hardware;
- load a version explicitly;
- verify the framework, CUDA runtime, driver, and compute capability together;
- do not assume the default is optimal for every GPU;
- preserve the module list in the run log.

Example:

```bash
module purge
module spider cuda
module load cuda/13.0     # Example only; verify for the target job
module list
nvidia-smi
nvcc --version
```

For Python frameworks that ship their own CUDA runtime, record the framework's view as well:

```bash
python - <<'PY'
try:
    import torch
    print("torch", torch.__version__)
    print("torch CUDA", torch.version.cuda)
    print("available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch inspection failed:", repr(exc))
PY
```

---

## 7. Slurm accounts, partitions, and resource eligibility

### 7.1 Accounts and partitions are distinct

A **Slurm account** identifies the department, class, or research group under which usage is charged and access is granted. A **partition** is a scheduling pool.

Typical syntax:

```bash
sbatch --account=eecs --partition=dgx2 job.sbatch
```

Do not place the ONID in `--account`. Discover the user's associations:

```bash
sacctmgr -nP show assoc where user="$USER" \
  format=Cluster,Account,User,Partition,DefaultQOS,QOS
```

If `sacctmgr` visibility is restricted, inspect a prior job:

```bash
sacct -u "$USER" -S now-30days \
  -o JobID,Account,Partition,State,Elapsed -X
```

### 7.2 Public/shared access

The FAQ says all users receive access to:

- `share`
- `dgx2`
- `dgxh`
- `preempt`

The partition table separately marks `ampere` as owner account `ALL`. This is an inconsistency. Test the live association and partition policy rather than assuming access.

### 7.3 Public partition map

The following names and ownership labels appear on the public Slurm page. Restricted pools can change as faculty-owned hardware is added or reassigned.

| Partition | Public owner/access label | Documented hardware/features |
|---|---|---|
| `share` | ALL | EL8/EL9; M60, A40; broad CPU-generation mix |
| `class` | class accounts | Class-reserved; varies |
| `dgxh` | ALL | EL9; H100, H200, VRAM feature classes; Sapphire Rapids/AVX-512 |
| `dgxh-ceoas` | `ceoas` | DGX H resources; H100/MIG-related features |
| `dgx2` | ALL | EL9; V100; Cascade Lake/AVX-512 |
| `gpu` | `eecs` | RTX 8000 |
| `gpu-dmv` | `dmv` | RTX 8000 |
| `ampere` | ALL in partition table | A40; EPYC |
| `athena` | `hlab` | A40; EPYC |
| `bee` | `fc-lab`, `sg-ecohydro` | V100 |
| `bee1` | `fc-lab` | V100 |
| `cbee` | `la-grp` | CPU |
| `cp-grp` | `cp-grp` | CPU |
| `ecohydro` | `sg-ecohydro` | CPU |
| `eecs` | `eecs` | RTX 2080/Broadwell |
| `eecs2` | `virl-grp` | RTX 6000 |
| `eecs3` | `rah-grp` | M60 |
| `hw-grp` | `hw-grp` | H100 |
| `maml` | `maml` | CPU |
| `mime1` | `kt-lab` | Group-owned |
| `mime2` | `jt-grp` | Group-owned |
| `mime3` | `mime3_grp` | Group-owned |
| `mime4` | `nrg` | Group-owned |
| `mime5` | `simon-grp` | Group-owned |
| `mime7` | `ba-grp` | Group-owned |
| `nacse` | `nacse` | A40, L40S |
| `nerhp` | `nse` | CPU |
| `nerhp2` | `ig-lab` | Group-owned |
| `nse3` | `cp-grp`, `tp-grp` | Group-owned |
| `sail` | `sail` | A40 |
| `soundbendor` | `soundbendor` | RTX 6000, T4 |
| `sy-grp` | `sy-grp` | H100, H200 |
| `tiamat` | `dmf`, `virl-grp`, `af-lab`, `iras` | L40S |
| `tp-grp` | `tp-grp` | Group-owned |
| `preempt` | ALL resources, low priority | Any eligible idle resources; subject to preemption |

List the current map:

```bash
scontrol show partition -o
sinfo -o '%P|%a|%l|%D|%F|%G'
```

### 7.4 Public usage limits

The public Slurm page documents resource-time limits for selected partitions. “Active resource-days” are aggregate concurrent consumption: two GPU-days can mean one GPU for two days or two GPUs for one day.

| Partition | Max walltime | Max CPUs/job | Active CPU-time/user | Max GPUs/job | Active GPU-time/user | Max RAM/job | Active RAM-time/user |
|---|---:|---:|---:|---:|---:|---:|---:|
| `share` | 7 days | 512 | 512 CPU-days | 2 | 2 GPU-days | 512 GB | 1,536 GB-days |
| `dgxh` | 2 days | 224 | 64 CPU-days | 8 | 2 GPU-days | 2,000 GB | 500 GB-days |
| `dgx2` | 7 days | 96 | 48 CPU-days | 16 | 8 GPU-days | 1,500 GB | 750 GB-days |
| `gpu` | 7 days | 40 | 48 CPU-days | 8 | 8 GPU-days | 750 GB | 750 GB-days |
| `ampere` | 2 days | 64 | 64 CPU-days | 2 | 2 GPU-days | 250 GB | 250 GB-days |
| `eecs` | 7 days | 40 | 48 CPU-days | 8 | 8 GPU-days | 250 GB | 250 GB-days |
| `preempt` | 7 days | Not stated | Not stated | Not stated | Not stated | Not stated | Not stated |

The page also documents per-user caps of 1,000 submitted jobs and 400 running jobs across partitions.

These are public defaults, not a substitute for current QOS and association limits. Query:

```bash
sacctmgr -nP show assoc where user="$USER" \
  format=Account,Partition,QOS,DefaultQOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmit
sacctmgr -nP show qos \
  format=Name,MaxWall,MaxJobsPU,MaxSubmitJobsPU,MaxTRESPU,GrpTRES,MaxTRESMinsPU
```

Field availability depends on the installed Slurm version and user permissions.

### 7.5 Defaults

The public examples identify:

- default partition: `share`;
- default walltime: 12 hours;
- default CPU/task count: 1;
- default task count: 1;
- default GPU count: 0.

Agent practice: do not rely on defaults in a production script. State all material resources.

---

## 8. Resource requisition protocol

### 8.1 Interactive allocation

Use an interactive allocation for diagnosis, compilation, small tests, profiler setup, or a single manual session:

```bash
srun \
  --partition=share \
  --time=01:00:00 \
  --cpus-per-task=4 \
  --mem=16G \
  --pty bash
```

GPU example:

```bash
srun \
  --partition=dgxh \
  --constraint=h100 \
  --gres=gpu:1 \
  --time=01:00:00 \
  --cpus-per-task=8 \
  --mem=64G \
  --pty bash
```

Preemptible example:

```bash
srun \
  --partition=preempt \
  --constraint=a40 \
  --gres=gpu:1 \
  --time=02:00:00 \
  --cpus-per-task=8 \
  --mem=64G \
  --pty bash
```

An interactive `srun` session is attached to the submit-side session. OSU warns that idle or resource-heavy submit-node `tmux` sessions can be terminated and that losing the submit session can terminate the interactive job. Use `sbatch` for long work.

### 8.2 Batch allocation

Submit a script:

```bash
mkdir -p logs
sbatch templates/cpu.sbatch
```

Capture the ID:

```bash
job_id=$(sbatch --parsable templates/cpu.sbatch)
printf 'submitted %s\n' "$job_id"
```

An agent should parse and preserve the job ID. It should not infer success from `sbatch` returning zero alone.

### 8.3 Resource request checklist

Specify:

- `--job-name`
- `--account` when a sponsored/private account is needed
- `--partition`
- `--time`
- `--nodes`
- `--ntasks` or `--ntasks-per-node` for distributed processes
- `--cpus-per-task` for threads used by one process
- `--mem` or `--mem-per-cpu`
- `--gres=gpu:N` for GPUs
- `--constraint` for required OS, CPU ISA, GPU model, or VRAM class
- `--output` and `--error`
- optional mail notification
- optional array bounds and concurrency cap
- optional signal before time limit for checkpointing

Do not request:

- all memory “just in case”;
- every core on a node when the code uses one;
- a premium GPU for CPU-only preprocessing;
- a seven-day walltime for a ten-minute test;
- a private partition without a valid account;
- an exact node unless there is a reproducible hardware reason.

Over-requesting can delay scheduling and blocks other users.

### 8.4 CPU semantics

For one multithreaded process:

```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
```

Then set the thread count:

```bash
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
```

For sixteen independent processes:

```bash
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
```

For MPI across two nodes:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
```

Do not confuse tasks with threads.

### 8.5 Memory semantics

`--mem=64G` is memory per allocated node. `--mem-per-cpu=4G` scales with allocated CPUs. Avoid setting both.

Use prior accounting to right-size:

```bash
sacct -j JOBID -X \
  -o JobID,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode
seff JOBID 2>/dev/null || true
```

`MaxRSS` for distributed or multi-step jobs may require examining `.batch`, `.extern`, and task steps, and is not always a perfect whole-job maximum.

### 8.6 Job arrays

Use arrays for homogeneous independent runs:

```bash
#SBATCH --array=0-999%20
```

The `%20` caps simultaneous array elements and reduces scheduler and filesystem pressure. Map `SLURM_ARRAY_TASK_ID` deterministically to a parameter row.

Do not submit thousands of one-second jobs. Bundle tiny tasks into one allocation.

### 8.7 Dependency chains

Slurm dependencies can encode a pipeline:

```bash
prep=$(sbatch --parsable prep.sbatch)
train=$(sbatch --parsable --dependency=afterok:"$prep" train.sbatch)
eval=$(sbatch --parsable --dependency=afterok:"$train" eval.sbatch)
printf '%s %s %s\n' "$prep" "$train" "$eval"
```

Use `afterok` for success-only transitions and save the graph in a run manifest.

### 8.8 `preempt` partition

`preempt` offers access to otherwise idle resources at low priority. A job may be canceled or preempted. It is appropriate when work is:

- restartable;
- checkpointed;
- idempotent;
- not deadline-critical;
- able to recover from termination.

Do not use it for an uncheckpointed multi-day run. Treat termination as normal, not exceptional.

---

## 9. Slurm command reference

### Queue and node state

```bash
squeue -u "$USER"
squeue -j JOBID
sinfo
sinfo -p dgxh
sinfo -N -o '%N|%P|%T|%c|%m|%G|%f'
scontrol show job JOBID
scontrol show node NODE
scontrol show partition PARTITION
```

OSU also documents aliases/utilities such as `sq`, `sqe`, `squ`, `showjob`, and `nodestat`; availability depends on the site's shell setup.

### Estimated start time

```bash
squeue -j JOBID --Format=starttime
showjob JOBID 2>/dev/null || true
```

An estimate can move as priorities, reservations, failures, and higher-priority jobs change.

### Cancel

```bash
scancel JOBID
scancel -u "$USER"        # destructive: cancels every job owned by the user
```

An agent must not run `scancel -u "$USER"` unless the human explicitly asks to cancel all jobs.

### Accounting

```bash
sacct -j JOBID -X \
  -o JobID,JobName,Account,Partition,State,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS,ExitCode
sstat -j JOBID.batch \
  --format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize
seff JOBID 2>/dev/null || true
```

### Update a pending job

OSU documents reducing a pending job's walltime, for example before maintenance:

```bash
scontrol update job JOBID TimeLimit=2-00:00:00
```

Not every field can be changed after submission. Cancel and resubmit when the resource shape is wrong.

### Pending reasons

Common reasons:

- `Resources`: matching resources are occupied.
- `Priority`: eligible but behind higher-priority work.
- `QOS...Limit`: a per-user, account, or QOS resource-time/job cap.
- `ReqNodeNotAvail, Reserved for maintenance`: requested walltime overlaps a reservation or node is unavailable.
- `MaxRunMins...`: active resource-time limit.
- `Dependency`: prerequisite has not satisfied the requested condition.
- `BadConstraints`: feature combination has no matching node.
- `InvalidAccount` or `InvalidQOS`: association or spelling problem.

Inspect the full reason:

```bash
squeue -j JOBID -o '%.18i %.12P %.16j %.8u %.2t %.10M %.10l %.6D %R'
scontrol show job JOBID
```

When asking support about a stuck job, leave it queued when practical and provide the job ID.

---

## 10. Storage and data lifecycle

### 10.1 Home directory

Documented fixed quota: **25 GB**.

Use home for:

- shell configuration;
- small source trees;
- SSH metadata;
- small logs and notebooks;
- lightweight configuration.

Do not place large environments, datasets, containers, model checkpoints, or build trees there. Exceeding home quota can break shell startup and Open OnDemand sessions.

Check:

```bash
quota -s 2>/dev/null || true
du -sh "$HOME" 2>/dev/null
```

`du` can be expensive on a very large tree; do not poll it continuously.

### 10.2 Global HPC share

Path:

```text
/nfs/hpc/share/ONID
```

A symlink may exist:

```text
~/hpc-share
```

The public FAQ says 1.5 TB, while the Optimize and Getting Started pages say 1 TB. This is a documented discrepancy. Determine the actual live quota with the site-provided `disk-usage` command, `quota`, or filesystem reporting.

Properties:

- visible on submit and compute nodes;
- high-speed shared scratch/medium-term workspace;
- not a backup;
- no snapshots or recovery for deleted files;
- may be subject to a 90-day purge policy;
- heavy metadata and small-file I/O can affect the whole cluster.

Use it for source, environments, datasets needed by multiple nodes, staged inputs, and output copied back from local scratch. Keep a separate durable copy elsewhere.

### 10.3 Local compute-node scratch

Every compute node has local `/scratch`; GPU-heavy systems may have terabytes of local SSD/NVMe. Some node-local filesystems are also exported under:

```text
/nfs/hpc/HOSTNAME
```

Properties:

- fastest choice for data-intensive temporary I/O;
- specific to a node;
- not backed up;
- may be purged without warning;
- unavailable or different when a job moves to another node;
- requires explicit stage-in and stage-out.

Safe pattern:

```bash
set -euo pipefail

shared_run="/nfs/hpc/share/$USER/runs/$SLURM_JOB_ID"
local_run="/scratch/$USER/$SLURM_JOB_ID"

mkdir -p "$shared_run" "$local_run"

cleanup() {
    status=$?
    # Copy valuable results before removing local data.
    rsync -a --partial "$local_run/output/" "$shared_run/output/" 2>/dev/null || true
    rm -rf -- "$local_run"
    exit "$status"
}
trap cleanup EXIT TERM INT

rsync -a --info=stats2 "$shared_run/input/" "$local_run/input/"
cd "$local_run"
srun ./program
```

Before using `rm -rf`, validate the path:

```bash
case "$local_run" in
  /scratch/"$USER"/*) ;;
  *) echo "Refusing unsafe cleanup path: $local_run" >&2; exit 2 ;;
esac
```

For multi-node jobs, `$SLURM_JOB_ID` scratch must be created separately on each node, usually with `srun --ntasks-per-node=1`.

### 10.4 Durable project and archival storage

OSU points users toward Engineering storage such as project storage and archival services rather than treating `hpc-share` as permanent. Confirm the current Depot, Guille, or Attic service applicable to the project and data classification.

Agent rule: never delete the only copy of an input, checkpoint, or result. A successful job exit is not proof that a result has been durably preserved.

### 10.5 Transfers

Documented options:

- SFTP/SCP to a submit node;
- MobaXterm, WinSCP, FileZilla, or Cyberduck;
- Open OnDemand Files for small transfers, documented up to 25 GB;
- SSHFS to expose `hpc-share` indirectly through a submit node.

Examples:

```bash
sftp ONID@submit.hpc.engr.oregonstate.edu
scp local-file ONID@submit.hpc.engr.oregonstate.edu:
rsync -avP local-dir/ ONID@submit.hpc.engr.oregonstate.edu:remote-dir/
```

Use VPN/campus network where required. Prefer `rsync --partial` for resumable large transfers. Verify checksums for important datasets.

### 10.6 Filesystem etiquette

In May 2026 OSU reported sustained heavy load and `hpc-share` pressure, including slow logins and environment activation. Administrators specifically discouraged continuous monitoring and recommended local `/scratch` for heavy AI/I/O work.

Agent practice:

- no `watch du ...`;
- no repeated recursive `find` across shared trees;
- no per-sample open/close pattern if data can be packed or cached;
- cap DataLoader worker counts after measurement;
- stage training shards to local NVMe;
- write logs in buffered batches;
- avoid creating millions of tiny files;
- avoid environment activation inside tight job loops;
- poll scheduler state at a measured interval, not every second.

---

## 11. Operating systems, modules, and software

### 11.1 Operating systems

OSU reports a mix of Rocky Linux 8 and Rocky Linux 9 nodes, approximately 60% EL8 and 40% EL9 at the time the inventory page was authored.

Select an OS feature when binary compatibility matters:

```bash
#SBATCH --constraint=el9
```

Do not build on EL9 and assume a dynamically linked executable will run on EL8. Containers can reduce user-space variation but do not replace the host kernel, GPU driver, or scheduler integration.

### 11.2 Lmod

Core commands:

```bash
module avail
module av python
module spider gcc
module load python/3.10
module list
module unload python
module purge
```

Compile-time and runtime module stacks must match. Record them:

```bash
module -t list 2>&1 | sort > "$run_dir/modules.txt"
```

A robust batch script uses a controlled module stack:

```bash
module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm
fi
module load gcc/VER
module load cuda/VER
```

On OSU, Slurm may itself be supplied by an automatically loaded module, so reload it after a purge when `srun` is no longer in `PATH`. Replace versions with live-discovered values. Avoid unversioned application modules for long-lived reproducibility unless the job also captures the resolved version.

If Slurm commands are missing, OSU says the `slurm` module should normally be loaded automatically. Try:

```bash
module load slurm
```

Persistent shell-path damage may require TEACH's **Reset Unix Config Files** function; back up custom dotfiles first.

### 11.3 Documented application stack

The public inventory lists:

| Software | Documented versions |
|---|---|
| MATLAB | 2023a, 2023b, 2024a, 2024b, 2025a, 2025b |
| Mathematica | 12.3, 13.3, 14.3 |
| Ansys | 2019r3, 2020r2, 2021r2, 2022r2, 2023r2, 2024r2, 2025r2 |
| Ansys Electronics Desktop | 2023r2, 2024r2 |
| StarCCM+ | 15.06, 16.06, 17.04, 18.06, 19.06 |
| Gurobi | 8.1, 9.1, 9.5, 10.0, 11.0, 12.0 |
| Intel oneAPI | 2022, 2024, 2025 |
| NVIDIA HPC SDK | 24.x, 25.x |
| GCC | 5.x through 15.2 |
| LLVM | 14 through 20 |
| CUDA | 11.x through 13.x |

This is not the complete module tree. Inventory it live:

```bash
module -t avail 2>&1 | less
module spider PACKAGE
```

Licenses may impose independent concurrency and eligibility limits. A Slurm allocation does not guarantee an application license.

### 11.4 Python virtual environments

Store environments on `hpc-share`, not home:

```bash
module purge
module load python/VER

python -m venv "/nfs/hpc/share/$USER/envs/project-py"
source "/nfs/hpc/share/$USER/envs/project-py/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

For reproducibility:

```bash
python -m pip freeze --all > environment.freeze.txt
python -VV > python-version.txt
```

Avoid running dozens of jobs that all import an environment with enormous numbers of tiny files directly from shared storage. Consider an Apptainer image or stage a packed environment to local scratch.

### 11.5 Conda/Miniforge

OSU recommends user-managed Conda environments and shows installing Miniforge under `/nfs/hpc/share/ONID`.

Conservative layout:

```text
/nfs/hpc/share/$USER/miniforge3
/nfs/hpc/share/$USER/conda-envs
/nfs/hpc/share/$USER/conda-pkgs
```

Configure:

```bash
export CONDA_ENVS_PATH="/nfs/hpc/share/$USER/conda-envs"
export CONDA_PKGS_DIRS="/nfs/hpc/share/$USER/conda-pkgs"
```

Use a lockfile where possible. Do not update a shared environment while jobs are using it; create an immutable versioned environment and switch a symlink only after validation.

### 11.6 Apptainer

OSU supports Apptainer. Store `.sif` images in `hpc-share`:

```bash
mkdir -p "/nfs/hpc/share/$USER/apptainer"
cd "/nfs/hpc/share/$USER/apptainer"

apptainer pull image.sif docker://REGISTRY/IMAGE:TAG
apptainer exec image.sif command
apptainer run image.sif
apptainer shell image.sif
```

A definition-file build may use:

```bash
apptainer build --fakeroot image.sif image.def
```

If a build is too resource-intensive or fails on a submit node, allocate a compute node first. Never assume a Docker command is available; use Apptainer's supported workflow.

GPU execution commonly needs `--nv`:

```bash
srun --gres=gpu:1 apptainer exec --nv image.sif python train.py
```

Verify that container CUDA libraries are compatible with the host driver.

Pin images by digest, not a floating `latest` tag, and store the definition file plus checksum:

```bash
sha256sum image.sif > image.sif.sha256
```

### 11.7 Compilers and MPI

Discover available MPI implementations:

```bash
module avail mpi
module spider openmpi
```

OSU's FAQ provides an Open MPI multi-node example and recommends preserving the compiler/MPI module pairing. A typical Slurm-native launch is:

```bash
module purge
module load gcc/VER
module load openmpi/VER
srun ./mpi-program
```

Some site builds may require `mpirun` with specific transport flags. Validate with a two-node smoke test and `/apps/samples` before scaling.

Never compile against one MPI and load another at runtime. Capture:

```bash
which mpicc
mpicc --showme 2>/dev/null || true
module list
ldd ./mpi-program
```

### 11.8 Requesting software

Try, in order:

1. check the normal path;
2. search Lmod;
3. use a user environment;
4. use Apptainer;
5. request a centrally managed package when it benefits the community or needs a license.

For a missing package, open an OSU support ticket with **COE HPC** in the subject. General COE software policy may require lead time, current supported Linux compatibility, an appropriate license, x86 support, active maintenance, and broad benefit. Export-controlled or restricted software requires IT involvement.

---

## 12. Reproducible batch design

A production batch script should produce a self-contained run record. At minimum, record:

```bash
date --iso-8601=seconds
hostname -f
id
pwd
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true
module -t list 2>&1 || true
env | grep '^SLURM_' | sort
ulimit -a
```

For GPU work:

```bash
nvidia-smi -L
nvidia-smi
```

For Python:

```bash
python -VV
python -m pip freeze --all
```

For compiled code:

```bash
ldd ./program
```

### 12.1 Run-directory convention

Recommended:

```text
project/
  src/
  config/
  scripts/
  env/
  runs/
    20260620T153000Z_git-abc1234_job-123456/
      manifest.txt
      modules.txt
      config.yaml
      stdout.log
      stderr.log
      checkpoints/
      output/
```

Do not write every job to a shared `output/` filename. Use `%j` or `%A_%a` in Slurm output paths.

### 12.2 Atomic outputs

Write to a temporary path, validate, then rename:

```bash
tmp="$run_dir/result.tmp"
final="$run_dir/result.parquet"

program --output "$tmp"
test -s "$tmp"
mv -f -- "$tmp" "$final"
```

A partially written file should never masquerade as a completed result.

### 12.3 Exit codes

Use:

```bash
set -euo pipefail
```

But understand its edge cases. Preserve the actual application exit status, and do not let a failing cleanup command overwrite it:

```bash
status=0
srun ./program || status=$?
rsync -a output/ "$shared_run/output/" || {
    echo "stage-out failed" >&2
    [ "$status" -ne 0 ] || status=90
}
exit "$status"
```

### 12.4 Signals and checkpoints

Ask Slurm to signal before walltime:

```bash
#SBATCH --signal=B:USR1@120
```

Trap it:

```bash
checkpoint() {
    echo "checkpoint requested at $(date --iso-8601=seconds)"
    # Replace with application-specific checkpoint action.
}
trap checkpoint USR1
```

A signal handler must be tested; it is not enough to include a trap that the application ignores.

---

## 13. Agent operating protocol

This section is the default contract for an autonomous coding agent on OSU HPC.

### 13.1 Human-controlled actions

The agent may prepare commands, scripts, and diagnostics. A human should control or explicitly authorize:

- initial authentication, Duo, VPN, and SSH host-key decisions;
- changes to credentials or SSH configuration;
- requests for new accounts, QOS, partitions, licenses, or software;
- submission of unusually large or long jobs;
- cancellation of all jobs;
- deletion of shared datasets or environments;
- publication or transfer of sensitive or regulated data;
- actions that may incur external charges.

### 13.2 Allowed submit-node work

Reasonable:

- `git status`, `git diff`, commit inspection;
- editing source and text files;
- modest dependency metadata operations;
- module and Slurm inspection;
- `sbatch`, `squeue`, `sacct`, `scontrol show`;
- checksums and small transfers;
- static analysis and lightweight tests.

Move to an allocation when an operation:

- sustains CPU use;
- consumes significant RAM;
- launches a compiler over a substantial project;
- opens many files;
- uses a GPU;
- runs a model, solver, simulation, notebook kernel, or database;
- preprocesses a dataset;
- creates an image;
- lasts more than a brief interactive check.

### 13.3 Forbidden or presumptively disallowed behavior

OSU account rules prohibit or restrict starting unapproved services and network scans. The agent must not:

- start a public web server, file-sharing service, IRC service, remote desktop server, or persistent listener;
- scan ports, hosts, subnets, or services;
- run cryptocurrency mining or unrelated distributed-computing workloads;
- bypass Slurm to consume compute nodes;
- use `sudo` or attempt privilege escalation;
- alter global software, `/apps`, Slurm configuration, modules, drivers, or system services;
- access another user's files or jobs;
- exfiltrate credentials or protected data;
- disable security controls;
- monopolize shared resources;
- run unattended heavy work on submit nodes.

Jupyter is acceptable only through the documented OnDemand workflow or an allocated compute node with a secure localhost tunnel. It must not be exposed to the public network.

### 13.4 Destructive-command guardrails

Before any deletion or overwrite:

1. print the resolved path;
2. verify it is inside the intended project/run root;
3. use `--` before path arguments;
4. use a dry run for `rsync --delete`;
5. preserve the only copy;
6. require human confirmation for broad deletion.

Never generate:

```bash
rm -rf "$some_maybe_empty_variable"
```

Use:

```bash
target=${target:?target must be set}
case "$(realpath -m "$target")" in
  "/nfs/hpc/share/$USER/project/"*) ;;
  *) echo "Refusing target outside project root" >&2; exit 2 ;;
esac
rm -rf -- "$target"
```

### 13.5 Secrets

- Do not put secrets in `#SBATCH` lines, command arguments, job names, output filenames, or Git.
- Prefer a protected file with mode 600 or an approved secret mechanism.
- Do not print the full environment.
- Redact tokens from diagnostic output.
- Assume Slurm output and project directories may be shared with collaborators.
- Avoid `set -x` when commands can contain credentials.

### 13.6 Minimal-resource progression

For a new workload:

1. syntax/static check locally or on submit;
2. one tiny CPU allocation;
3. one tiny GPU allocation if needed;
4. one representative input;
5. inspect runtime, RSS, GPU memory, utilization, and I/O;
6. set a realistic safety margin;
7. submit a bounded pilot;
8. scale via array or distributed run;
9. inspect accounting and tune.

Do not begin with the maximum partition limit.

### 13.7 Polling and observation

A scheduler agent should poll no more frequently than necessary—typically every 30–60 seconds for short tests and several minutes for long jobs. It should stop polling when the job reaches a terminal state.

Avoid continuous filesystem monitoring. Prefer:

```bash
squeue -h -j "$job_id" -o '%T'
```

After completion:

```bash
sacct -j "$job_id" -X -n -P -o State,ExitCode,Elapsed,MaxRSS
```

### 13.8 Completion criteria

A job is not “done” merely because it left `squeue`. The agent must check:

- terminal Slurm state is `COMPLETED`;
- `ExitCode` is successful;
- expected files exist and are nonempty;
- validation tests pass;
- stage-out completed;
- manifest and logs were saved;
- no local-only valuable data remains;
- downstream dependency or report was updated.

States such as `OUT_OF_MEMORY`, `TIMEOUT`, `PREEMPTED`, `NODE_FAIL`, `CANCELLED`, and `FAILED` require explicit handling.

### 13.9 Handling ambiguity

When the public guide and live cluster disagree, preserve evidence:

```bash
date --iso-8601=seconds
scontrol show partition PARTITION
scontrol show node NODE
module -t avail 2>&1
```

Then favor the live scheduler and current admin instructions. Do not “repair” system configuration. For an access or configuration issue, prepare a support request containing:

- ONID;
- submit host;
- date/time and timezone;
- job ID;
- exact command;
- exact error;
- partition/account;
- relevant `scontrol show job` output;
- concise expected versus observed behavior;
- subject containing `COE HPC`.

---

## 14. Codex integration

Codex reads `AGENTS.md` before work and layers instructions from the Codex home directory and repository hierarchy. The included `AGENTS.md` is designed to fit under Codex's default project-instruction size limit while this full guide remains a linked reference.

Recommended placement:

```text
~/.codex/AGENTS.md              # reusable OSU HPC safety defaults
project/AGENTS.md               # repository-specific build/test/run rules
project/subsystem/AGENTS.md     # narrower overrides if needed
```

Keep the root file focused on:

- where work may occur;
- how tests and jobs are launched;
- resource ceilings;
- environment/module requirements;
- data locations;
- forbidden actions;
- success criteria.

Do not paste this entire guide into every repository. Codex's documented default combined project instruction limit is 32 KiB. Put operational specifics close to the code they govern.

### Suggested Codex session sequence

1. Human connects to VPN and SSH.
2. Start Codex from the repository root on a submit node.
3. Ask Codex to summarize active `AGENTS.md` instructions.
4. Run the read-only inventory script.
5. Let Codex inspect code and prepare a minimal job.
6. Review `git diff` and the `sbatch` script.
7. Submit a smoke test.
8. Let Codex inspect `sacct`, logs, and outputs.
9. Scale only after the pilot meets acceptance criteria.

Do not grant a coding agent blanket approval for destructive shell commands or unrestricted network activity merely because it is running in a research account.

---

## 15. Standard templates

This package includes:

- `templates/cpu.sbatch`
- `templates/gpu.sbatch`
- `templates/array.sbatch`
- `templates/mpi.sbatch`
- `templates/preempt-checkpoint.sbatch`

Each template is intentionally conservative and contains placeholders. Validate partition, account, module version, executable, and paths before submission.

### Submission preflight

```bash
bash -n templates/gpu.sbatch
grep '^#SBATCH' templates/gpu.sbatch
mkdir -p logs
sbatch --test-only templates/gpu.sbatch 2>/dev/null || true
```

`--test-only` support and output vary by Slurm version. It does not validate application logic or license availability.

---

## 16. Troubleshooting

### 16.1 Job is pending

```bash
squeue -j JOBID -o '%.18i %.12P %.16j %.2t %.10M %.10l %.6D %R'
scontrol show job JOBID
```

Remedies by cause:

- `Resources`: wait, reduce request, broaden compatible partitions, or choose a more available GPU class.
- `Priority`: wait; do not repeatedly cancel and resubmit without reason.
- QOS/resource-time limit: allow active jobs to finish or reduce concurrency.
- maintenance reservation: reduce walltime to fit, or wait.
- bad constraints: inspect live feature spelling.
- invalid account: use a valid association or contact support.

### 16.2 Out of memory

Symptoms: `OUT_OF_MEMORY`, kernel kill, exit 137, or OSU `tracejob`/Slurm record.

Inspect:

```bash
sacct -j JOBID -X -o JobID,State,ReqMem,MaxRSS,Elapsed,ExitCode
seff JOBID 2>/dev/null || true
```

Fix the cause before blindly increasing memory:

- reduce batch size or workers;
- eliminate duplicate in-memory copies;
- use streaming/chunking;
- inspect CPU versus GPU OOM separately;
- set worker limits;
- request measured memory plus margin.

### 16.3 Time limit

State: `TIMEOUT`.

Actions:

- inspect actual progress;
- checkpoint;
- optimize or split the task;
- request a realistic walltime within the partition limit;
- avoid using maximum walltime when shorter jobs schedule sooner.

### 16.4 Preemption

State: `PREEMPTED`, `CANCELLED`, or site-specific termination record.

- resume from latest validated checkpoint;
- make stage-out idempotent;
- keep checkpoint cadence proportional to expected loss;
- consider a non-preempt partition for a final run.

### 16.5 GPU is not visible

Check allocation:

```bash
echo "$SLURM_JOB_ID"
echo "${CUDA_VISIBLE_DEVICES-}"
scontrol show job "$SLURM_JOB_ID" | grep -i -E 'gres|tres'
nvidia-smi -L
```

Likely causes:

- requested only a feature, not `--gres`;
- running on submit node;
- container lacks `--nv`;
- framework installed CPU-only;
- driver/runtime mismatch;
- MIG/GRES naming changed.

### 16.6 CUDA/framework mismatch

Record:

```bash
module list
nvidia-smi
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

Then align:

- host driver;
- loaded toolkit if compilation is required;
- framework wheel/container runtime;
- GPU architecture;
- extension build cache.

Delete only the project-specific stale extension cache, not shared system caches.

### 16.7 Illegal instruction

Likely cause: compiled for a newer ISA than the scheduled CPU.

- inspect `--constraint`;
- rebuild with a conservative target;
- compile and run on the same feature class;
- avoid `-march=native` unless every run uses the same compatible constraint.

### 16.8 Environment activation is slow

- `hpc-share` may be under metadata load;
- reduce package/file count;
- use a packed environment or Apptainer;
- avoid repeated activation inside loops;
- stage read-heavy environment data to local scratch;
- do not hammer the filesystem with diagnostic scans.

### 16.9 Disk quota or no space

```bash
quota -s 2>/dev/null || true
df -hT "$HOME" "/nfs/hpc/share/$USER" /scratch 2>/dev/null || true
disk-usage 2>/dev/null || true
```

Clean only known regenerable files. Do not assume `df` reflects a per-user quota on a shared filesystem.

### 16.10 Direct SSH denied

`pam_slurm_adopt` denial means there is no matching allocation on the target node. Start from a submit host and request resources through Slurm.

### 16.11 Submit session terminated

Possible causes include submit-node CPU or memory limits, idle policy, or maintenance. Move long work to `sbatch`. An existing batch job should continue even when the client disconnects.

### 16.12 Node failure

State may be `NODE_FAIL` or application I/O errors.

- preserve logs;
- inspect `sacct`;
- do not pin to the failed node;
- resubmit from a checkpoint;
- report persistent hardware symptoms with job/node IDs.

---

## 17. Performance and fairness

### CPU

- benchmark thread scaling; more cores can be slower;
- respect NUMA effects for memory-bound programs;
- use `srun --cpu-bind` only after understanding the application's topology;
- avoid oversubscription from OpenMP + BLAS + DataLoader worker multiplication.

### GPU

- confirm utilization and memory use during a pilot;
- use the least scarce suitable GPU;
- do not reserve a GPU while performing long CPU-only preprocessing;
- release interactive allocations when idle;
- OSU has warned that unused GPU allocations may be terminated, especially on premium GPU partitions.

### I/O

- stage hot data to local scratch;
- use larger sequential reads/writes;
- aggregate small outputs;
- avoid excessive stat calls;
- checkpoint to local scratch then copy validated checkpoints to shared storage.

### Scheduler

- arrays with concurrency caps are better than job floods;
- accurate walltime can improve fit;
- flexible constraints schedule more easily;
- exact `--nodelist` requests should be exceptional;
- `preempt` can improve throughput for restartable work without blocking owners.

---

## 18. Live inventory and reconciliation

Run:

```bash
bash bin/osu-hpc-inventory.sh
```

The report captures:

- identity and host;
- OS and Slurm versions;
- accounts and QOS visible to the user;
- partition definitions;
- node state, cores, memory, GRES, and features;
- current jobs;
- module inventory;
- filesystem mounts and quotas;
- GPU details only when already inside a GPU allocation.

It is read-only and deliberately avoids recursive scans or fan-out SSH.

Review `LIVE_INVENTORY_CHECKLIST.md` to update the guide's assumptions.

### Known public-document discrepancies to reconcile

1. DGX-2 count: four in one summary, five in detailed/boilerplate text, and `dgx2-3` reported dead in March 2026.
2. `cn-v-1`–`cn-v-9`: nine-host range versus “8x” model count.
3. Several `cn-e*` and `cn-a*` ranges versus model-count labels.
4. `cn-x-1`: publicly labeled H100 with 140 GB.
5. `hpc-share` quota: 1 TB on Optimize/Getting Started versus 1.5 TB in FAQ.
6. Universal access: FAQ names four partitions; partition table also labels `ampere` as ALL.
7. OnDemand hostname: both `ondemand...` and `submit...` appear.
8. GPU/MIG GRES names and availability after the June 2026 Slurm and driver updates.
9. Current DGX/EL9 restoration state after maintenance.
10. Aggregate GPU count versus listed active hosts.

These discrepancies do not make the public pages unusable; they mean an agent must not convert them into hard-coded assumptions.

---

## 19. Support escalation template

```text
Subject: COE HPC — [short problem] — job [JOBID]

ONID:
Date/time with timezone:
Submit host:
Job ID:
Account:
Partition:
Requested resources:
Node(s), if allocated:
Exact command:
Exact error:
Expected behavior:
Observed behavior:
Relevant output:
  scontrol show job JOBID
  sacct -j JOBID -X -o JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode
  module -t list
Steps already tried:
Impact/deadline:
```

Do not include passwords, access tokens, private keys, or restricted data.

---

## 20. Source index

Official OSU sources reviewed 2026-06-20:

1. HPC home and eligibility:  
   https://it.engineering.oregonstate.edu/hpc
2. Getting started:  
   https://it.engineering.oregonstate.edu/hpc/getting-started
3. Hardware and software inventory:  
   https://it.engineering.oregonstate.edu/hpc/about-cluster
4. Slurm usage, partitions, features, and limits:  
   https://it.engineering.oregonstate.edu/hpc/slurm-howto
5. FAQ, access, storage, jobs, Jupyter, and Apptainer:  
   https://it.engineering.oregonstate.edu/hpc/faqs
6. Environment, storage, transfer, and OnDemand guidance:  
   https://it.engineering.oregonstate.edu/hpc/optimize
7. Lmod guidance:  
   https://it.engineering.oregonstate.edu/hpc/lmod-howto
8. Current status and historical operational notices:  
   https://it.engineering.oregonstate.edu/hpc/hpc-cluster-status-and-news
9. COE account rules:  
   https://it.engineering.oregonstate.edu/what-are-rules-and-guidelines-using-my-account
10. COE Unix workstation/server guidance:  
    https://it.engineering.oregonstate.edu/what-are-rules-and-guidelines-unix-workstations-and-servers

Codex instruction behavior:

11. OpenAI, “Custom instructions with AGENTS.md”:  
    https://developers.openai.com/codex/guides/agents-md

---

## 21. Compact operational checklist

Before work:

```text
[ ] VPN/gateway and submit login are valid.
[ ] Current status page and `sinfo` are checked.
[ ] Slurm account, partition, QOS, and limits are known.
[ ] Storage quota and backup destination are known.
[ ] Modules/environment are pinned.
[ ] The smallest representative test is defined.
```

Before `sbatch`:

```text
[ ] No heavy work occurs on submit.
[ ] CPU/task/thread semantics are correct.
[ ] Memory and walltime are measured estimates.
[ ] GPU is explicitly requested, not merely constrained.
[ ] Logs include job IDs and do not expose secrets.
[ ] Local scratch has stage-in/stage-out and cleanup.
[ ] Preemptible work has a tested checkpoint path.
[ ] Output paths are unique and atomic.
```

After completion:

```text
[ ] `sacct` state and exit code are checked.
[ ] Outputs pass validation.
[ ] Important data is copied off local scratch.
[ ] Run manifest, modules, code revision, and configuration are saved.
[ ] Resource efficiency is reviewed before scaling.
[ ] Allocations and interactive sessions are released.
```
