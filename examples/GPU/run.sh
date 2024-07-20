#!/bin/bash
#SBATCH -n 1
#SBATCH -t 20:1:00
#SBATCH --mem-per-cpu=20000
#SBATCH --exclude=milan[8,10,23]
#SBATCH --gpus=1



module load cuda/12.2.1
rm *.OUT
module load scicomp-python-env
#srun python main.py
srun /appl/scibuilder-mamba/aalto-rhel9/prod/software/scicomp-python-env/2024-01/f56a564/bin/python main.py

