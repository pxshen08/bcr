#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=ProtT5
#SBATCH --mem-per-cpu=50G
#SBATCH --time=2-00:00:00
#SBATCH --error=log/ProtT5_embed.%A_%a.err
#SBATCH --output=log/ProtT5_embed.%A_%a.out
#SBATCH --array=1-4

#module load miniconda
conda activate base

script="/home/mist/projects/Wang2023/scripts/5_ProtT5_embed.py"
path="/home/mist/projects/Wang2023/data/FASTA"
outpath="/home/mist/projects/Wang2023/data/BCR_embed/datap"

files=("combined_distinct_heavy.fa"
       "combined_distinct_light.fa"
       "combined_cdr3_heavy.fa" 
       "combined_cdr3_light.fa"
       "Bcell.fa")

taskID=${SLURM_ARRAY_TASK_ID}

echo "[$0 $(date +%Y%m%d-%H%M%S)] [start] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"

file=${files[$taskID-1]}
outfile="${file%.*}_ProtT5.pkl"

echo python $script $path/$file $outpath/$outfile
python $script $path/$file $outpath/$outfile

echo "[$0 $(date +%Y%m%d-%H%M%S)] [end] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"
