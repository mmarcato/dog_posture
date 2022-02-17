#!/bin/sh

# Slurm flags
#SBATCH -p DevQ
#SBATCH -N 1
#SBATCH -t 00:60:00
#SBATCH --job-name=MM-Simple-SKB-RF

# Charge job to myproject
#SBATCH -A tieng028c

# Write to file
#SBATCH -o src/models/output/MM-Simple-SKB-RF.txt

# Mail me on job start & end
#SBATCH --mail-user=marinara.marcato@tyndall.ie
#SBATCH --mail-type=BEGIN,END

cd $SLURM_SUBMIT_DIR

echo $GAUSS_SCRDIR

module load conda/2
source activate /ichec/work/tieng028c/venv

python3 src/models/1.0-mm-main.py
