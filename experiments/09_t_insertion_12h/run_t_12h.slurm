#!/bin/bash
#SBATCH --cluster=chip-gpu
#SBATCH --job-name=fcnv2_t_12h
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/fcn_inference/t_12h/logs/%x_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/fcn_inference/t_12h/logs/%x_%j.err
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --account=pi_halem
#SBATCH --partition=gpu

mkdir -p /umbc/rs/pi_halem/users/yashd1/fcn_inference/t_12h/logs

source /umbc/rs/pi_halem/users/yashd1/envs/fcn3_env/bin/activate
export PYTHONPATH=/umbc/rs/pi_halem/users/yashd1/fcn_inference/t_12h/OSSE-FourCastNetv2:${PYTHONPATH}
export PYTHONUNBUFFERED=1

python /umbc/rs/pi_halem/users/yashd1/fcn_inference/t_12h/inf.py
