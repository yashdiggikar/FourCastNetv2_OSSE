#!/bin/bash
#SBATCH --cluster=chip-gpu
#SBATCH --job-name=era5_rmse_dl
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data/logs/download-%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/AURORA_DATA/rmse_data/logs/download-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

export USER="${USER:-yashd1}"
export RESEARCH_ROOT="/umbc/rs/pi_halem/users/${USER}"

export HOME="${RESEARCH_ROOT}"
export XDG_CACHE_HOME="${RESEARCH_ROOT}/.cache"
export TMPDIR="${RESEARCH_ROOT}/.tmp/${SLURM_JOB_ID}"
export PYTHONUSERBASE="${RESEARCH_ROOT}/.local"
export PATH="${RESEARCH_ROOT}/.local/bin:${PATH}"

mkdir -p "$TMPDIR" \
  "${RESEARCH_ROOT}/AURORA_DATA/rmse_data/logs"

VENV="${RESEARCH_ROOT}/venvs/aurora310"
source "${VENV}/bin/activate"

echo "========================================"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Started : $(date)"
echo "Python  : $(which python)  $(python --version 2>&1)"
echo "========================================"

python "${RESEARCH_ROOT}/AURORA_DATA/rmse_data/download_era5.py"

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
