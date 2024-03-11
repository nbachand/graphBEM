#!/bin/bash
#SBATCH --job-name=BEM_MC
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --time=48:00:00
#SBATCH --mem 128G
#SBATCH --partition=serc
#SBATCH --mail-type=ALL

conda activate pycascade
export PYTHONUNBUFFERED=True
python runMyBuildingMC.py
