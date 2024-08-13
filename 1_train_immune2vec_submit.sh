#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=immune2vec
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=50G
#SBATCH --time=1-00:00:00
#SBATCH --error=log/immune2vec.%A_%a.err
#SBATCH --output=log/immune2vec.%A_%a.out
#SBATCH --array=1-20

#module load miniconda
conda activate base

arguments=("H FULL 25" "L FULL 25" "H CDR3 25" "L CDR3 25"
           "H FULL 50" "L FULL 50" "H CDR3 50" "L CDR3 50"
           "H FULL 100" "L FULL 100" "H CDR3 100" "L CDR3 100"
           "H FULL 150" "L FULL 150" "H CDR3 150" "L CDR3 150"
           "H FULL 200" "L FULL 200" "H CDR3 200" "L CDR3 200")

taskID=${SLURM_ARRAY_TASK_ID}

argument=${arguments[$taskID-1]}

echo "[$0 $(date +%Y%m%d-%H%M%S)] [start] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"

echo $argument

python /home/mist/projects/Wang2023/scripts/1_train_immune2vec.py $argument

echo "[$0 $(date +%Y%m%d-%H%M%S)] [end] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"
