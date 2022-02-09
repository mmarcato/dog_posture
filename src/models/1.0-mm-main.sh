#!/bin/sh

# Slurm flags
#SBATCH -p DevQ
#SBATCH -N 4
#SBATCH -t 00:10:00
#SBATCH --job-name=MM-main-1

# Charge job to myproject
#SBATCH -A tieng028c

# Write stdout+stderr to file
#SBATCH -o output.txt

# Mail me on job start & end
#SBATCH --mail-user=marinara.marcato@tyndall.ie
#SBATCH --mail-type=BEGIN,END

cd $SLURM_SUBMIT_DIR

echo $GAUSS_SCRDIR

module load conda/2
source activate /ichec/work/tieng028c/venv

python3 hello.py
