#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=baseline
#SBATCH --mem-per-cpu=50G
#SBATCH --time=2-00:00:00
#SBATCH --error=log/baseline_embed.%A_%a.err
#SBATCH --output=log/baseline_embed.%A_%a.out
#SBATCH --array=1-4

#module load miniconda
#conda activate immune2vec

script="/home/mist/projects/Wang2023/scripts/6_baseline_embed.py"
path="/home/mist/projects/Wang2023/data/FASTA"
outpath="/home/mist/projects/Wang2023/data/BCR_embed/datab"
embedding="physicochemical"

train=( "combined_cdr3_light.fa"
       "combined_distinct_light.fa"
       "Bcell.fa"
       "combined_distinct_heavy.fa"
       "Bcell.fa"
       "combined_cdr3_heavy.fa"
       "Bcell.fa")

taskID=3 #${SLURM_ARRAY_TASK_ID}

echo "[$0 $(date +%Y%m%d-%H%M%S)] [start] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"

trainfile=${train[$taskID-1]}
outfile="${trainfile%.*}_${embedding}.pkl"

echo python $script $embedding $path/$trainfile $outpath/$outfile
python $script $embedding $path/$trainfile $outpath/$outfile

echo "[$0 $(date +%Y%m%d-%H%M%S)] [end] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"
