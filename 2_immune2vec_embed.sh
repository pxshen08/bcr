#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=immune2vec
#SBATCH --mem-per-cpu=50G
#SBATCH --time=5:00:00
#SBATCH --error=log/immune2vec_embed.%A_%a.err
#SBATCH --output=log/immune2vec_embed.%A_%a.out
#SBATCH --array=1-20

module load miniconda
conda activate immune2vec

script="/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/scripts/2_immune2vec_embed.py"
path="/gpfs/ysm/project/kleinstein/mw957/repos/bcr_embeddings/data"
outpath="/gpfs/gibbs/pi/kleinstein/mw957/data/BCR_embed/data"
models=("H_FULL_25" "L_FULL_25" "H_CDR3_25" "L_CDR3_25"
        "H_FULL_50" "L_FULL_50" "H_CDR3_50" "L_CDR3_50"
        "H_FULL_100" "L_FULL_100" "H_CDR3_100" "L_CDR3_100"
        "H_FULL_150" "L_FULL_150" "H_CDR3_150" "L_CDR3_150"
        "H_FULL_200" "L_FULL_200" "H_CDR3_200" "L_CDR3_200")
files=("combined_distinct_heavy.fa" "combined_distinct_light.fa" "combined_cdr3_heavy.fa" "combined_cdr3_light.fa"
       "combined_distinct_heavy.fa" "combined_distinct_light.fa" "combined_cdr3_heavy.fa" "combined_cdr3_light.fa"
       "combined_distinct_heavy.fa" "combined_distinct_light.fa" "combined_cdr3_heavy.fa" "combined_cdr3_light.fa"
       "combined_distinct_heavy.fa" "combined_distinct_light.fa" "combined_cdr3_heavy.fa" "combined_cdr3_light.fa"
       "combined_distinct_heavy.fa" "combined_distinct_light.fa" "combined_cdr3_heavy.fa" "combined_cdr3_light.fa")

taskID=${SLURM_ARRAY_TASK_ID}

echo "[$0 $(date +%Y%m%d-%H%M%S)] [start] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"

file=${files[$taskID-1]}
model=${models[$taskID-1]}
outfile="${file%.*}_immune2vec_${model}.pkl"

echo python $script $model $path/$file $outpath/$outfile
python $script $model $path/$file $outpath/$outfile

echo "[$0 $(date +%Y%m%d-%H%M%S)] [end] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"
