import argparse
import pickle
import sys

codedir = "/home/mist/projects/Wang2023/immune2vec_model/embedding"
sys.path.append(codedir)

from . import sequence_modeling
#import immune2vec

from Bio import SeqIO
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("model", type=str, help="model")
parser.add_argument("fasta_file", type=str, help="Path to the fasta file")
parser.add_argument("output_file", type=str, help="Output file path")

args = parser.parse_args()

# Load model
model_path = "/gpfs/gibbs/pi/kleinstein/mw957/data/BCR_embed/model/immune2vec/"
model = sequence_modeling.load_protvec(model_path + args.model + ".immune2vec")

# Load fasta file
ids = []
seqs = []
for seq_record in SeqIO.parse(args.fasta_file, "fasta"):
    ids.append(seq_record.id)
    seqs.append(''.join(seq_record.seq))
    
seqs = pd.Series(seqs, index = ids)
print(f"Read {args.fasta_file} with {len(seqs)} sequences")

# Embed the sequences
embedded = immune2vec.embed_data(model, seqs)

# Save as a pickled data frame
embedded.to_pickle(args.output_file)
print(f"Saved {args.output_file} with {embedded.shape[0]} sequences")