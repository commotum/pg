#!/usr/bin/env bash
#SBATCH --job-name=g2b-release-32768
#SBATCH --partition=share
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/release-32768-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/goal2-phase4-benchmarks/release-32768-%j.err

set -euo pipefail

cd /nfs/hpc/share/peterj29/pg/src/pg
VOCABS=${VOCABS:-32768}
SUBMIT=${SUBMIT:-1}
export VOCABS SUBMIT

bash goal-2/4-submit-benchmark-matrix.sh
