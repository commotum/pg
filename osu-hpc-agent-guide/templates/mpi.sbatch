#!/usr/bin/env bash
#SBATCH --job-name=osu-mpi-smoke
#SBATCH --partition=share
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=osu-mpi-%j.out
#SBATCH --error=osu-mpi-%j.err

# This template deliberately requests two nodes and therefore should be reviewed
# before an agent submits it. Confirm the live MPI/compiler module pairing:
#   module spider openmpi
#
# Replace the commented module versions with a site-supported pair.
# Submit a topology smoke test as-is only after review:
#   sbatch mpi.sbatch
#
# Or run a compiled MPI program:
#   sbatch mpi.sbatch ./my-mpi-program args...

set -euo pipefail
umask 077

module purge
if ! command -v srun >/dev/null 2>&1; then
    module load slurm
fi
# module load gcc/VERSION
# module load openmpi/VERSION

submit_dir=${SLURM_SUBMIT_DIR:-$PWD}
if [[ -d /nfs/hpc/share/$USER ]]; then
    shared_root=${RUN_ROOT:-/nfs/hpc/share/$USER/agent-runs}
else
    shared_root=${RUN_ROOT:-$submit_dir/agent-runs}
fi
run_dir="$shared_root/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$run_dir"

{
    echo "job_id=$SLURM_JOB_ID"
    echo "nodes=$SLURM_JOB_NUM_NODES"
    echo "tasks=$SLURM_NTASKS"
    echo "node_list=$SLURM_JOB_NODELIST"
    echo "git_commit=$(git -C "$submit_dir" rev-parse HEAD 2>/dev/null || true)"
    echo
    module -t list 2>&1 || true
    echo
    command -v mpicc 2>/dev/null || true
    mpicc --showme 2>/dev/null || true
} >"$run_dir/manifest.txt"

if [[ $# -gt 0 ]]; then
    srun "$@"
else
    # Confirms one task per rank and the distribution across allocated nodes.
    srun bash -lc 'printf "rank=%s/%s host=%s\n" \
      "$SLURM_PROCID" "$SLURM_NTASKS" "$(hostname)"' \
      | sort >"$run_dir/ranks.txt"
fi
