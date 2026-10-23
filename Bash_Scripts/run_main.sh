#!/usr/local/bin/bash

# FILENAME: run_main.sh
#SBATCH -A rothte -p cpu

#SBATCH --nodes=1 --ntasks=32 --time=20:00

module reset
module load anaconda

cd $SLURM_SUBMIT_DIR
hostname

conda activate singEM
python main.py -f ./Meshes/surfPart/surfPart_FD.json

