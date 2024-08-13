#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=esm2
#SBATCH --mem-per-cpu=50G
#SBATCH --time=2-00:00:00
#SBATCH --error=log/esm2_embed.%A_%a.err
#SBATCH --output=log/esm2_embed.%A_%a.out
#SBATCH --array=1-3


#module load miniconda
conda activate base

script="/home/mist/projects/Wang2023/scripts/3_esm_embed.py"

path="/home/mist/projects/Wang2023/data/FASTA"
outpath="/home/mist/projects/Wang2023/data/BCR_embed/datae"

files=("combined_distinct_heavy.fa" 
       "combined_distinct_light.fa" 
       "combined_cdr3_heavy.fa" 
       "combined_cdr3_light.fa"
       "ellebedy_heavy.fa"
       "Bcell.fa"
       "Bcell_fixed.fa")

taskID=${SLURM_ARRAY_TASK_ID}

echo "[$0 $(date +%Y%m%d-%H%M%S)] [start] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"

file=${files[$taskID-1]}
outfile="${file%.*}.pt"

echo python $script $path/$file $outpath/$outfile
python $script $path/$file $outpath/$outfile

echo "[$0 $(date +%Y%m%d-%H%M%S)] [end] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"
