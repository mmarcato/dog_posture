#!/bin/sh

# Slurm flags
#SBATCH -p ProdQ
#SBATCH -N 1
#SBATCH -t 5:00:00
#SBATCH --job-name=MM-4-imbalanced-2

# Charge job to myproject
#SBATCH -A tieng028c

# Write to file
#SBATCH -o 4-imbalanced-2.txt

# Mail me on job start & end
#SBATCH --mail-user=marinara.marcato@tyndall.ie
#SBATCH --mail-type=BEGIN,END

cd $SLURM_SUBMIT_DIR

echo $GAUSS_SCRDIR

module load conda/2
source activate /ichec/work/tieng028c/venv

python3 4-imbalanced-2.py
