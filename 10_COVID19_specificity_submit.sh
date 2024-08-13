#!/bin/bash
#SBATCH --partition=scavenge
#SBATCH --job-name=COVID19_random
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks=4 --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --error=log/COVID19_random.%A_%a.err
#SBATCH --output=log/COVID19_random.%A_%a.out
#SBATCH --array=45-52

conda activate base

embeddings=("immune2vec_H_FULL_25 HL" "immune2vec_H_FULL_50 HL" "immune2vec_H_FULL_100 HL" "immune2vec_H_FULL_150 HL" "immune2vec_H_FULL_200 HL" "immune2vec_H_FULL_500 HL" "immune2vec_H_FULL_1000 HL" 
            "immune2vec_H_FULL_25 H" "immune2vec_H_FULL_50 H" "immune2vec_H_FULL_100 H" "immune2vec_H_FULL_150 H" "immune2vec_H_FULL_200 H" "immune2vec_H_FULL_500 H" "immune2vec_H_FULL_1000 H"
            "immune2vec_H_CDR3_25 HL" "immune2vec_H_CDR3_50 HL" "immune2vec_H_CDR3_100 HL" "immune2vec_H_CDR3_150 HL" "immune2vec_H_CDR3_200 HL" "immune2vec_H_CDR3_500 HL" "immune2vec_H_CDR3_1000 HL"
            "immune2vec_H_CDR3_25 H" "immune2vec_H_CDR3_50 H" "immune2vec_H_CDR3_100 H" "immune2vec_H_CDR3_150 H" "immune2vec_H_CDR3_200 H" "immune2vec_H_CDR3_500 H" "immune2vec_H_CDR3_1000 H"
            "frequency_FULL HL" "frequency_FULL H" "frequency_CDR3 HL" "frequency_CDR3 H"
            "physicochemical_FULL HL" "physicochemical_FULL H" "physicochemical_CDR3 HL" "physicochemical_CDR3 H"
            "esm2_FULL HL" "esm2_FULL H" "esm2_CDR3 HL" "esm2_CDR3 H" 
            "ProtT5_FULL HL" "ProtT5_FULL H" "ProtT5_CDR3 HL" "ProtT5_CDR3 H"
            "esm2_3B_FULL HL" "esm2_3B_FULL H" "esm2_3B_CDR3 HL" "esm2_3B_CDR3 H"
            "antiBERTy_FULL HL" "antiBERTy_FULL H"  "antiBERTy_CDR3 H" "antiBERTy_CDR3 HL" "antiBERTy_CDR3_ELL HL" "antiBERTy_cdr3_specificity_ELL HL" "antiBERTy_CDR3 HL"
            "antiBERTy_Bcell_1 H")

taskID=${SLURM_ARRAY_TASK_ID}

echo "[$0 $(date +%Y%m%d-%H%M%S)] [start] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"

embedding=${embeddings[$taskID-1]}

echo python 10_COVID19_specificity.py $embedding
python 10_COVID19_specificity.py $embedding

echo "[$0 $(date +%Y%m%d-%H%M%S)] [end] $SLURM_JOBID $SLURM_ARRAY_TASK_ID"