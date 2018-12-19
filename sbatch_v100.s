#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH -c8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --job-name=v100-512-0.001
source ${HOME}/.bashrc
python3 ${HOME}/IMGCaptioning/main.py --encoder_model_path="/scratch/dd2645/cv-project/v100-512-0.001/encoder-21.ckpt" --decoder_model_path="/scratch/dd2645/cv-project/v100-512-0.001/decoder-21.ckpt" --saving_model_path="/scratch/dd2645/cv-project/v100-512-0.001"
