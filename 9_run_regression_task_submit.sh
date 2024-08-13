#!/bin/bash
#SBATCH --partition=scavenge
#SBATCH --job-name=junction_random
#SBATCH --ntasks=4 --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=05:00:00
#SBATCH --error=log/regression_random.%A_%a.err
#SBATCH --output=log/regression_random.%A_%a.out
#SBATCH --array=49-52
# 45,47

module load miniconda
conda activate r_seurat

# change to array jobs for junction predictions
tasks=("mu_H immune2vec_H_FULL_25" "mu_H immune2vec_H_FULL_50" "mu_H immune2vec_H_FULL_100" "mu_H immune2vec_H_FULL_150" "mu_H immune2vec_H_FULL_200" "mu_H immune2vec_H_FULL_500" "mu_H immune2vec_H_FULL_1000"
       "mu_L immune2vec_L_FULL_25" "mu_L immune2vec_L_FULL_50" "mu_L immune2vec_L_FULL_100" "mu_L immune2vec_L_FULL_150" "mu_L immune2vec_L_FULL_200" "mu_L immune2vec_L_FULL_500" "mu_L immune2vec_L_FULL_1000"
       "Jlen_H immune2vec_H_FULL_25" "Jlen_H immune2vec_H_FULL_50" "Jlen_H immune2vec_H_FULL_100" "Jlen_H immune2vec_H_FULL_150" "Jlen_H immune2vec_H_FULL_200" "Jlen_H immune2vec_H_FULL_500" "Jlen_H immune2vec_H_FULL_1000"
       "Jlen_L immune2vec_L_FULL_25" "Jlen_L immune2vec_L_FULL_50" "Jlen_L immune2vec_L_FULL_100" "Jlen_L immune2vec_L_FULL_150" "Jlen_L immune2vec_L_FULL_200" "Jlen_L immune2vec_L_FULL_500" "Jlen_L immune2vec_L_FULL_1000"
       "mu_H physicochemical" "mu_L physicochemical" "Jlen_H physicochemical" "Jlen_L physicochemical"
       "mu_H frequency" "mu_L frequency" "Jlen_H frequency" "Jlen_L frequency"
       "mu_H esm2" "mu_L esm2" "Jlen_H esm2" "Jlen_L esm2"
       "mu_H ProtT5" "mu_L ProtT5" "Jlen_H ProtT5" "Jlen_L ProtT5"
       "mu_H esm2_3B" "mu_L esm2_3B" "Jlen_H esm2_3B" "Jlen_L esm2_3B"
       "mu_H antiBERTy" "mu_L antiBERTy" "Jlen_H antiBERTy" "Jlen_L antiBERTy")

taskID=${SLURM_ARRAY_TASK_ID}

echo "[$0 $(date +%Y%m%d-%H%M%S)] [start] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"

task=${tasks[$taskID-1]}

echo /gpfs/gibbs/project/kleinstein/mw957/conda_envs/r_seurat/bin/python 7_run_regression_task.py $task
/gpfs/gibbs/project/kleinstein/mw957/conda_envs/r_seurat/bin/python 7_run_regression_task.py $task

echo "[$0 $(date +%Y%m%d-%H%M%S)] [end] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"
