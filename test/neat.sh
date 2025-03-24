#!/bin/bash --login
#SBATCH --job-name=bp_neat
#SBATCH --output=bp_neat%j.out
#SBATCH --error=bp_neat%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:59:00

# Load conda environment (adjust the path if necessary)
source ~/miniforge3/etc/profile.d/conda.sh
conda activate backprop-neat
export PATH=~/miniforge3/envs/kanji-gpu/bin:$PATH


# Run the full bp-neat.py script
python bp-neat.py