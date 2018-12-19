#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=p40_4,p100_4,v100_sxm2_4,v100_pci_2
#SBATCH -c8
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --job-name=512-0.001
source ${HOME}/.bashrc
python3 ${HOME}/IMGCaptioning/main.py --saving_model_path="/scratch/dd2645/cv-project/512-0.001"
