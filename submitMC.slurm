#!/bin/bash
#SBATCH --job-name=BEM_MC
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --time=1:00:00
#SBATCH --mem 128G
#SBATCH --partition=serc
#SBATCH --mail-type=ALL

export PYTHONUNBUFFERED=True
python runMyBuildingMC.py