#!/usr/bin/env bash
#SBATCH --job-name=pg-phase1-smoke
#SBATCH --partition=gpu
#SBATCH --constraint=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/phase1-smoke/slurm-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/phase1-smoke/slurm-%j.err

set -euo pipefail
umask 077

RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/phase1-smoke}
ENV_PATH=${ENV_PATH:-/nfs/hpc/share/peterj29/pg/envs/pg-py311-torch-smoke-20260621}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
JOB_DIR="$RUN_ROOT/$SLURM_JOB_ID"

mkdir -p "$JOB_DIR"
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.out" "$JOB_DIR/slurm.out" || true
ln -sf "$RUN_ROOT/slurm-$SLURM_JOB_ID.err" "$JOB_DIR/slurm.err" || true

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm/current 2>/dev/null || module load slurm 2>/dev/null || true
fi
module load cuda/13.0 2>/dev/null || module load cuda/12.8 2>/dev/null || true
source "$ENV_PATH/bin/activate"

{
    echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
    echo "job_id=$SLURM_JOB_ID"
    echo "job_name=${SLURM_JOB_NAME-}"
    echo "host=$(hostname -f 2>/dev/null || hostname)"
    echo "partition=${SLURM_JOB_PARTITION-}"
    echo "nodes=${SLURM_JOB_NODELIST-}"
    echo "cpus_per_task=${SLURM_CPUS_PER_TASK-}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES-}"
    echo "repo_path=$REPO_PATH"
    echo "git_commit=$(git -C "$REPO_PATH" rev-parse HEAD 2>/dev/null || true)"
    echo
    echo "[modules]"
    module -t list 2>&1 || true
    echo
    echo "[slurm]"
    env | grep '^SLURM_' | sort
} >"$JOB_DIR/manifest.txt" 2>&1

{
    echo "[nvidia-smi -L]"
    nvidia-smi -L
    echo
    echo "[query]"
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
    echo
    echo "[full]"
    nvidia-smi
    echo
    echo "[nvcc]"
    nvcc --version 2>/dev/null || true
} >"$JOB_DIR/gpu.txt" 2>&1

{
    echo "[python]"
    command -v python
    python -VV
    echo
    echo "[pip freeze]"
    python -m pip freeze --all
} >"$JOB_DIR/python.txt" 2>&1

python - <<'PY' >"$JOB_DIR/torch-smoke.txt" 2>&1
import json
import torch

result = {
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
}

if not result["cuda_available"]:
    raise SystemExit(f"CUDA unavailable: {json.dumps(result, sort_keys=True)}")

if result["cuda_device_count"] != 1:
    raise SystemExit(f"Expected exactly one CUDA device: {json.dumps(result, sort_keys=True)}")

device_name = torch.cuda.get_device_name(0)
result["device_name"] = device_name

torch.manual_seed(1)
x = torch.randn(512, 512, device="cuda", dtype=torch.float32, requires_grad=True)
y = (x @ x.T).square().mean()
y.backward()
torch.cuda.synchronize()

result["loss"] = float(y.detach().cpu())
result["grad_norm"] = float(x.grad.norm().detach().cpu())
result["max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
result["status"] = "ok"

print(json.dumps(result, indent=2, sort_keys=True))
PY

echo "complete" >"$JOB_DIR/COMPLETE.txt"
