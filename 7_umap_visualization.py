import sys
import argparse
import torch
from umap import UMAP
import pandas as pd

parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("in_file", type=str, help="Path to the fasta file")
parser.add_argument("output_file", type=str, help="Output file path")
args = parser.parse_args()

# Load embedding file
if "pt" in args.in_file:
    data = torch.load(args.in_file).numpy()
elif "pkl" in args.in_file:
    data = pd.read_pickle(args.in_file).values
print(f"Read {args.in_file} with {data.shape[0]} sequences")

#data= torch.tensor(data, dtype=torch.float32).to("cuda")
# Embed the sequences using tsne
tsne_df = pd.DataFrame(UMAP().fit_transform(data))

# Save as a pickled data frame
tsne_df.to_pickle(args.output_file)
print(f"Saved {args.output_file} with {tsne_df.shape[0]} sequences")
