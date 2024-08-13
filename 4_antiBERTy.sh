#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a5000
#SBATCH --job-name=antiBERTy
#SBATCH --mem-per-cpu=32G
#SBATCH --time=02:00:00
#SBATCH --error=log/antiBERTy_embed.%A_%a.err
#SBATCH --output=log/antiBERTy_embed.%A_%a.out
#SBATCH --array=2

#module load miniconda
conda activate base

script="/home/mist/projects/Wang2023/scripts/4_antiBERTy.py"
path="/home/mist/projects/Wang2023/data/FASTA"
outpath="/home/mist/projects/Wang2023/data/BCR_embed/dataa"

files=("combined_distinct_heavy.fa" 
       "combined_distinct_light.fa" 
       "combined_cdr3_heavy.fa" 
       "combined_cdr3_light.fa"
       "30_1.fa"
       "ellebedy_light.fa"
       "ellebedy_heavy.fa"
        "Bcell.fa")

taskID=${SLURM_ARRAY_TASK_ID}

echo "[$0 $(date +%Y%m%d-%H%M%S)] [start] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"

file=${files[$taskID-1]}
outfile="${file%.*}_antiBERTy.pt"

#echo /gpfs/gibbs/project/kleinstein/mw957/conda_envs/torch/bin/python $script $path/$file $outpath/$outfile
#/gpfs/gibbs/project/kleinstein/mw957/conda_envs/torch/bin/python $script $path/$file $outpath/$outfile
echo python $script $path/$file $outpath/$outfile
python $script  $path/$file $outpath/$outfile

echo "[$0 $(date +%Y%m%d-%H%M%S)] [end] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"