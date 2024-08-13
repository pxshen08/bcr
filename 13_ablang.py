import argparse
import torch
import numpy as np
import time
from Bio import SeqIO
import pandas as pd
import math
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from ablang1 import pretrained


def batch_loader(data, batch_size):
    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        yield i, end_idx, data[i:end_idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Input path")
    parser.add_argument("--fasta_file", default="/home/mist/projects/Wang2023/data/FASTA/combined_cdr3_heavy.fa", type=str,
                        help="Path to the fasta file")
    parser.add_argument("--output_file", default="/home/mist/projects/Wang2023/data/BCR_embed/datai/combined_cdr3_heavy_ablang.pt",
                        type=str, help="Output file path")
    args = parser.parse_args()
    print(args)
    return args


def preprocess_sequence(sequence):
    valid_characters = "MRHKDESTNQCGPAVIFYWL*"
    return ''.join([char if char in valid_characters else '*' for char in sequence])


args = parse_args()

# 加载预训练模型
MODEL_LOCATION = "heavy"  # 可以是 "heavy" 或 "light"
ablang_model = pretrained(MODEL_LOCATION,model_folder="/home/mist/projects/Wang2023/scripts/ablang1/model-weights-heavy/",device='cuda')
ablang_model.freeze()

sequences = []
for seq_record in SeqIO.parse(args.fasta_file, "fasta"):
    preprocessed_seq = preprocess_sequence(str(seq_record.seq))
    sequences.append(preprocessed_seq)
print(f"Read {args.fasta_file} with {len(sequences)} sequences")

start_time = time.time()
batch_size = 32
n_seqs = len(sequences)
dim = 768  # AbLang的嵌入维度
n_batches = math.ceil(n_seqs / batch_size)
embeddings = torch.empty((n_seqs, dim))

# Use the batch_loader function to iterate through batches
i = 1
for start, end, batch in batch_loader(sequences, batch_size):
    print(f'Batch {i}/{n_batches}\n')

    tokens = ablang_model.tokenizer(batch, pad=True)
    if torch.cuda.is_available():
        tokens = tokens.to(device="cuda", non_blocking=True)

    with torch.no_grad():
        output = ablang_model.AbRep(tokens)
        x = output.last_hidden_states

    x = x.mean(dim=1)  # 计算每个序列的平均表示
    embeddings[start:end] = x.cpu()

    # 额外计算
    x1 = x
    x2 = nn.Linear(dim, 1).cuda()(x1)
    criterion_A = nn.MSELoss()
    max_length = max(len(string) for string in batch)
    padded_ascii_list = [torch.tensor([ord(char) / 255.0 for char in string] + [0.0] * (max_length - len(string)),
                                      dtype=torch.float) for string in batch]
    tensor = pad_sequence(padded_ascii_list, batch_first=True)
    b2 = nn.Linear(tensor.size(1), 1)(tensor).cuda()
    loss_A = criterion_A(b2, x2).cuda()
    print(loss_A)
    i += 1

end_time = time.time()
print(end_time - start_time)

torch.save(embeddings, args.output_file)
