from antiberty import AntiBERTyRunner
import argparse
import torch
import numpy as np
import time
from Bio import SeqIO
import pandas as pd
import math
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def batch_loader(data, batch_size):
    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        yield i, end_idx, data[i:end_idx]

# parser = argparse.ArgumentParser(description="Input path")
# parser.add_argument("fasta_file", type=str, help="Path to the fasta file")
# parser.add_argument("output_file", type=str, help="Output file path")
# args = parser.parse_args()
def parse_args():
 parser = argparse.ArgumentParser(description="Input path")
 parser.add_argument("--fasta_file",default="/home/mist/projects/Wang2023/data/FASTA/cdr3seq.fasta",type=str, help="Path to the fasta file")
 parser.add_argument("--output_file",default="/home/mist/projects/Wang2023/data/BCR_embed/dataa/cdr3seq_antiBERTy.pt", type=str, help="Output file path")
 args = parser.parse_args()
 print(args)
 return args
args=parse_args()#原先没有20240126

antiberty = AntiBERTyRunner()
sequences = []
for seq_record in SeqIO.parse(args.fasta_file, "fasta"):
    sequences.append(''.join(seq_record.seq))
print(f"Read {args.fasta_file} with {len(sequences)} sequences")

start_time = time.time()
batch_size = 32
n_seqs = len(sequences)
dim = 512
n_batches = math.ceil(n_seqs / batch_size)
embeddings = torch.empty((n_seqs, dim))

# Use the batch_loader function to iterate through batches
i = 1
for start, end, batch in batch_loader(sequences, batch_size):
    print(f'Batch {i}/{n_batches}\n')
    x = antiberty.embed(batch)
    x = [a.mean(axis = 0) for a in x]
    embeddings[start:end] = torch.stack(x)

    x1=torch.stack(x)
    x2 = nn.Linear(512, 1).cuda()(x1)
    criterion_A =nn.MSELoss()
    max_length = max(len(string) for string in batch)
    # 将每个字符串转换为其 ASCII 值，并进行填充并归一化处理
    padded_ascii_list = [torch.tensor([ord(char) / 255.0 for char in string] + [0.0] * (max_length - len(string)),
                                      dtype=torch.float) for string in batch]
    # 构建张量
    tensor = pad_sequence(padded_ascii_list, batch_first=True)
    b2 = nn.Linear(tensor.size(1), 1)(tensor).cuda()
    loss_A = criterion_A(b2, x2).cuda()
    print(loss_A)
    i += 1

end_time = time.time()
print(end_time - start_time)

torch.save(embeddings, args.output_file)