#!/bin/bash
#SBATCH --cluster=chip-gpu
#SBATCH --job-name=fcnv2_tuv_global
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/fcn_inference/tuv_global/logs/%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/fcn_inference/tuv_global/logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --account=pi_halem
#SBATCH --partition=gpu

mkdir -p /umbc/rs/pi_halem/users/yashd1/fcn_inference/tuv_global/logs

export HOME=/umbc/rs/pi_halem/users/yashd1
export PYTHONUNBUFFERED=1

source /umbc/rs/pi_halem/users/yashd1/envs/fcn3_env/bin/activate

export PYTHONPATH=/umbc/rs/pi_halem/users/yashd1/fcn_inference/tuv_global/OSSE-FourCastNetv2:${PYTHONPATH}

python /umbc/rs/pi_halem/users/yashd1/fcn_inference/tuv_global/inf.py
