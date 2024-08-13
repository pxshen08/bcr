import argparse
import sys
#codedir = "/gpfs/ysm/project/mw957/repos/bcr_embeddings/embedding"
#sys.path.append(codedir)
import ngram
from Bio import SeqIO
import pandas as pd
import numpy as np
import peptides
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# https://pypi.org/project/peptides/

parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("embedding", type=str, help="Type of baseline embedding for the amino acids")
parser.add_argument("fasta_file", type=str, help="Path to the train fasta file")
parser.add_argument("output_file", type=str, help="Output file path")
args = parser.parse_args()

def read_fasta(path):
    ids = []
    seqs = []
    for seq_record in SeqIO.parse(path, "fasta"):
        ids.append(seq_record.id)
        seqs.append(''.join(seq_record.seq))
    seqs = pd.Series(seqs, index = ids)
    print(f"Read {path} with {len(seqs)} sequences")
    return seqs

# Load fasta file
seqs = read_fasta(args.fasta_file)

# Embed the sequences
if args.embedding == "TFIDF":
    vectorizer = ngram.fit_ngrams_TFIDF(seqs)
    embedded = ngram.transform_ngrams_TFIDF(vectorizer, seqs)
elif args.embedding == "physicochemical":
    embedded = pd.DataFrame([ peptides.Peptide(s).descriptors() for s in seqs ])
    embedded.index = seqs.index
elif args.embedding == "frequency":
    embedded = pd.DataFrame([ProteinAnalysis(s).count_amino_acids() for s in seqs])
    length = np.sum(embedded, axis = 1)
    embedded = np.apply_along_axis(lambda x: x/length, 0, embedded)
    embedded = pd.DataFrame(embedded)
    embedded.index = seqs.index

# Save as a pickled data frame
embedded.to_pickle(args.output_file)
print(f"Saved {args.output_file} with {embedded.shape[0]} sequences")