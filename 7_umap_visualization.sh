#!/bin/bash
#SBATCH --partition=bigmem
#SBATCH --job-name=umap
#SBATCH --mem-per-cpu=400G
#SBATCH --time=2-00:00:00
#SBATCH --error=log/tSNE_embed.%A_%a.err
#SBATCH --output=log/tSNE_embed.%A_%a.out
#SBATCH --array=1-4

module load miniconda
conda activate base

script="/home/mist/projects/Wang2023/scripts/7_umap_visualization.py"

path="/home/mist/projects/Wang2023/data/BCR_embed/dataa/"

files=(    "combined_distinct_light_ProtT5.pkl" "combined_cdr3_heavy_ProtT5.pkl" "combined_distinct_heavy_ProtT5.pkl"
"ellebedy_heavy_antiBERTy.pt" "ellebedy_light_antiBERTy.pt" "Bcell_antiBERTy.pt" "Bcell_3_ablang.pt" "combined_cdr3_heavy_ablang.pt" "combined_distinct_light_ablang.pt" "Bcell_ablang.pt"
 "combined_distinct_light_physicochemical.pkl" "cc_encoder_outputslstm.pt" "cdr3b_antiBERTy.pt" "cdr3seq_antiBERTy.pt")

taskID=${SLURM_ARRAY_TASK_ID}

echo "[$0 $(date +%Y%m%d-%H%M%S)] [start] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"

file=${files[$taskID-1]}
outfile="${file%.*}_umap.pkl"

echo python $script $path/$file $path/$outfile
python $script $path/$file $path/$outfile
echo "[$0 $(date +%Y%m%d-%H%M%S)] [end] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"
