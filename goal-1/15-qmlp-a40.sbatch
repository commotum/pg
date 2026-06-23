#!/usr/bin/env bash
#SBATCH --job-name=pg-p15-qbase
#SBATCH --partition=share
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/nfs/hpc/share/peterj29/pg/runs/phase15-record-qmlp-sp16384-baseline/slurm-%j.out
#SBATCH --error=/nfs/hpc/share/peterj29/pg/runs/phase15-record-qmlp-sp16384-baseline/slurm-%j.err

set -euo pipefail
umask 077

VOCAB_SIZE=${VOCAB_SIZE:-16384}
SEED_VALUE=${SEED_VALUE:?SEED_VALUE is required}
REPO_PATH=${REPO_PATH:-/nfs/hpc/share/peterj29/pg/src/pg}
CASEOPS_ROOT=${CASEOPS_ROOT:-/nfs/hpc/share/peterj29/pg/data-exports/caseops-sp${VOCAB_SIZE}}
CASEOPS_OUT=${CASEOPS_OUT:-$CASEOPS_ROOT/datasets}
DATASET_NAME=${DATASET_NAME:-fineweb10B_sp${VOCAB_SIZE}_lossless_caps_caseops_v1_reserved}

export REPO_PATH
export VOCAB_SIZE
export SEED_VALUE
export CASEOPS_ROOT
export DATA_PATH=${DATA_PATH:-$CASEOPS_OUT/datasets/$DATASET_NAME}
export TOKENIZER_PATH=${TOKENIZER_PATH:-$CASEOPS_OUT/tokenizers/fineweb_${VOCAB_SIZE}_bpe_lossless_caps_caseops_v1_reserved.model}
export RUN_ROOT=${RUN_ROOT:-/nfs/hpc/share/peterj29/pg/runs/phase15-record-qmlp-sp${VOCAB_SIZE}-baseline}

exec bash "$REPO_PATH/goal/14-qmlp-a40.sbatch"
