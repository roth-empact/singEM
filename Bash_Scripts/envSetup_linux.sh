#!/usr/local/bin/bash

# FILENAME: envSetup_linux.sh
#SBATCH -A [ACCOUNT-NAME]

#SBATCH --nodes=1 --ntasks=64 --time=4:00:00 -p cpu

module reset
module load anaconda

cd $SLURM_SUBMIT_DIR

conda install anaconda
conda env create -f ./Bash_Scripts/singEM_linux.yml
