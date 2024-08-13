#!/bin/bash
#SBATCH --partition=week
#SBATCH --job-name=genes_random
#SBATCH --ntasks=4 --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=3-00:00:00
#SBATCH --error=log/classification_random.%A_%a.err
#SBATCH --output=log/classification_random.%A_%a.out
#SBATCH --array=73
# 73-78
# 67,68,71

# non-random 28,34-35,41-42
# random 21,34-35,41-42
# 6-7,13-14,20-21,27-28,34-35,41-42
# esm2_3b 67-72
#module load miniconda
conda activate base

script="/home/mist/projects/Wang2023/scripts/8_run_classification_task2.py"

classificationTasks=("VH immune2vec_H_FULL_25" "VH immune2vec_H_FULL_50" "VH immune2vec_H_FULL_100" "VH immune2vec_H_FULL_150" "VH immune2vec_H_FULL_200" "VH immune2vec_H_FULL_500" "VH immune2vec_H_FULL_1000"
                     "JH immune2vec_H_FULL_25" "JH immune2vec_H_FULL_50" "JH immune2vec_H_FULL_100" "JH immune2vec_H_FULL_150" "JH immune2vec_H_FULL_200" "JH immune2vec_H_FULL_500" "JH immune2vec_H_FULL_1000"
                     "VL immune2vec_L_FULL_25" "VL immune2vec_L_FULL_50" "VL immune2vec_L_FULL_100" "VL immune2vec_L_FULL_150" "VL immune2vec_L_FULL_200" "VL immune2vec_L_FULL_500" "VL immune2vec_L_FULL_1000"
                     "JL immune2vec_L_FULL_25" "JL immune2vec_L_FULL_50" "JL immune2vec_L_FULL_100" "JL immune2vec_L_FULL_150" "JL immune2vec_L_FULL_200" "JL immune2vec_L_FULL_500" "JL immune2vec_L_FULL_1000"
                     "isoH immune2vec_H_FULL_25" "isoH immune2vec_H_FULL_50" "isoH immune2vec_H_FULL_100" "isoH immune2vec_H_FULL_150" "isoH immune2vec_H_FULL_200" "isoH immune2vec_H_FULL_500" "isoH immune2vec_H_FULL_1000" 
                     "isoL immune2vec_L_FULL_25" "isoL immune2vec_L_FULL_50" "isoL immune2vec_L_FULL_100" "isoL immune2vec_L_FULL_150" "isoL immune2vec_L_FULL_200" "isoL immune2vec_L_FULL_500" "isoL immune2vec_L_FULL_1000"
                     "VH physicochemical" "JH physicochemical" "VL physicochemical" "JL physicochemical"  "isoH physicochemical" "isoL physicochemical"
                     "VH frequency" "JH frequency" "VL frequency" "JL frequency"  "isoH frequency" "isoL frequency"
                     "VH esm2" "JH esm2" "VL esm2" "JL esm2" "isoH esm2" "isoL esm2" 
                     "VH ProtT5" "JH ProtT5" "VL ProtT5" "JL ProtT5"  "isoL ProtT5"
                     "VH esm2_3B" "JH esm2_3B" "VL esm2_3B" "JL esm2_3B" "isoH esm2_3B" "isoL esm2_3B"
                     "isoL antiBERTy" "VH frequency" "isoH antiBERTy" "JH antiBERTy"  "VH antiBERTy" "VL antiBERTy" "JL antiBERTy" "JL ProtT5" "JL ablang" )

taskID=${SLURM_ARRAY_TASK_ID}

echo "[$0 $(date +%Y%m%d-%H%M%S)] [start] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"

task=${classificationTasks[$taskID-1]}

echo python 8_run_classification_task2.py $task
python 8_run_classification_task2.py $task

echo "[$0 $(date +%Y%m%d-%H%M%S)] [end] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"