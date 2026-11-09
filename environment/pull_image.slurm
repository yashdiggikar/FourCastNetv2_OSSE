#!/bin/bash
#SBATCH --cluster=chip-gpu
#SBATCH --job-name=pull_modulus
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/fcn_inference/logs/pull_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/fcn_inference/logs/pull_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=pi_halem
#SBATCH --partition=gpu

mkdir -p /umbc/rs/pi_halem/users/yashd1/fcn_inference/logs

module load apptainer

export APPTAINER_CACHEDIR=/umbc/rs/pi_halem/users/yashd1/.apptainer_cache
mkdir -p "${APPTAINER_CACHEDIR}"

echo "Pulling modulus:23.08 image..."
apptainer pull \
    /umbc/rs/pi_halem/users/yashd1/fcn_inference/modulus_23.08.sif \
    docker://nvcr.io/nvidia/modulus/modulus:23.08

echo "Done. SIF at: /umbc/rs/pi_halem/users/yashd1/fcn_inference/modulus_23.08.sif"
ls -lh /umbc/rs/pi_halem/users/yashd1/fcn_inference/modulus_23.08.sif
