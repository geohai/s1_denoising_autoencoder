#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=smem
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --job-name=autoencoder
#SBATCH --mail-user=benjamin.lucas@colorado.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=model.%j.out
#SBATCH --error=model.%j.err

module purge
module load python/3.6.5
module load gcc/6.1.0
source /curc/sw/anaconda3/latest
conda activate mypyenv

python3 /projects/belu5721/s1_denoise_autoencoder/code/main_rc.py
