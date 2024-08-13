import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import argparse
import numpy as np

#Input_path="/home/mist/projects/Wang2023/data/FASTA"
#output_file="/home/mist/projects/Wang2023/data/BCR_embed/data"
#fasta_file=("combined_distinct_heavy.fa"
      # "combined_distinct_light.fa"
       #"combined_cdr3_heavy.fa"
       #"combined_cdr3_light.fa")

parser = argparse.ArgumentParser(description="Input path")
parser.add_argument("fasta_file", type=str, help="Path to the fasta file")
parser.add_argument("output_file", type=str, help="Output file path")
args = parser.parse_args()

MODEL_LOCATION = "esm2_t33_650M_UR50D"
TOKS_PER_BATCH = 4096
REPR_LAYERS = [-1]

model, alphabet = pretrained.load_model_and_alphabet(MODEL_LOCATION)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    print("Transferred model to GPU")

dataset = FastaBatchedDataset.from_file(args.fasta_file)
batches = dataset.get_batch_indices(TOKS_PER_BATCH, extra_toks_per_seq=1)
data_loader = torch.utils.data.DataLoader(
    dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
)

print(f"Read {args.fasta_file} with {len(dataset)} sequences")

assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in REPR_LAYERS)
repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in REPR_LAYERS]

mean_representations = []
seq_labels = []

with torch.no_grad():
    for batch_idx, (labels, strs, toks) in enumerate(data_loader):
        print(
            f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
        )
        if torch.cuda.is_available():
            toks = toks.to(device="cuda", non_blocking=True)

        out = model(toks, repr_layers=repr_layers, return_contacts=False)

        representations = {
            layer: t.to(device="cpu") for layer, t in out["representations"].items()
        }

        for i, label in enumerate(labels):
            seq_labels.append(label)
            mean_representation = [t[i, 1 : len(strs[i]) + 1].mean(0).clone()
                    for layer, t in representations.items()]
            mean_representations.append(mean_representation[0])
            
mean_representations = torch.vstack(mean_representations)
ordering = np.argsort([int(i) for i in seq_labels])
mean_representations = mean_representations[ordering,:]
torch.save(mean_representations, args.output_file)
