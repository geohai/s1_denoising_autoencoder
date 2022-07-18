#!/bin/bash
#SBATCH --partition=aa100-ucb
#SBATCH --job-name=alpine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --mail-user=benjamin.lucas@colorado.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=alpine.%j.out
#SBATCH --error=alpine.%j.err

module purge
source /curc/sw/anaconda3/latest
conda activate mypyenv_alpine

python3 /projects/belu5721/s1_denoise_autoencoder/code/main_rc_alpine.py -D 8192
