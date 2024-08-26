#!/bin/bash
#SBATCH --job-name=gdino_sam_waterbird_mask
#SBATCH --output=/home/mila/j/jaewoo.lee/logs/gdino_sam_waterbird_mask_%j.out
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32Gb
#SBATCH --partition=long
#SBATCH --mail-user=jaewoo.lee@mila.quebec
#SBATCH --mail-type=ALL

module load miniconda/3
conda activate dinosam

cd /home/mila/j/jaewoo.lee/projects/GroundedDINO_SAM

python grounded_dino_sam_on_waterbird.py
