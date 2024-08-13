##-----------------------------
## 
## Script name: train_immune2vec.py
##
## Purpose of script: Script to train immune2vec on four single-cell BCR datasets
##
## Author: Mamie Wang
##
## Date Created: 2022-08-18
##
## Email: mamie.wang@yale.edu
##
## ---------------------------
## load the packages and inputs

import pickle
import sys

sys.path.append("/home/mist/projects/Wang2023/immune2vec_model/embedding")

import sequence_modeling,generate_model
# codedir = "/gpfs/ysm/project/mw957/repos/bcr_embeddings/embedding"
# sys.path.append(codedir)
#import immune2vec

import pandas as pd
import numpy as np

import argparse

def parse_args():
 parser = argparse.ArgumentParser(description="Gene usage tasks")
 parser.add_argument('--chain', default='L',type=str, help="Chain type (H, L)")
 parser.add_argument('--input', default='CDR3',type=str, help="Input sequence (FULL or CDR3 or FULL_CDR3)")
 parser.add_argument('--dim',default='25', type=int, help="Latent dimesnion of immune2vec")
 args = parser.parse_args()
 print(args)
 return args
##-----------------------------
## Load data
base_dir = "/home/mist/projects/Wang2023/data/Csv/"
out_dir = "/home/mist/projects/Wang2023/data/BCR_embed/dataa/"
args=parse_args()#原先没有20240201

if args.chain == "H":
    if args.input == "CDR3":
        sequence_train = pd.read_csv(base_dir + "combined_cdr3_heavy.csv").SequenceID#heavy_cdr3
    elif args.input == "FULL":
        sequence_train = pd.read_csv(base_dir + "combined_distinct_heavy1.csv").heavy
elif args.chain == "L": 
    if args.input == "CDR3":
        sequence_train = pd.read_csv(base_dir + "combined_cdr3_light.csv").SequenceID#light_cdr3
    elif args.input == "FULL":
        sequence_train = pd.read_csv(base_dir + "combined_distinct_light1.csv").light

##-----------------------------
## Check input
seq_len = sequence_train.apply(len)

print("# sequences: " + str(sequence_train.shape[0]))

lengths = sequence_train.apply(len)
print("Length range: " + str(min(lengths)) + " - " + str(max(lengths)) + " AA.")

##-----------------------------
## Run immune2vec
out_id = args.chain + "_" + args.input + "_" + str(args.dim)
out_corpus_fname = out_dir + out_id

model = generate_model.main(sequence_train, out_corpus_fname, args.dim)
filename = out_corpus_fname + ".immune2vec"
#generate_model.enerate_model_exec(model, filename)

print("Immune2vec model saved at: " + filename)