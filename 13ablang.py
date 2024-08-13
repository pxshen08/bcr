import os
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
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def batch_loader(data, batch_size):
    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        yield i, end_idx, data[i:end_idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Input path")
    parser.add_argument("--fasta_file", default="/home/mist/projects/Wang2023/data/FASTA/combined_distinct_light.fa", type=str,
                        help="Path to the fasta file")
    parser.add_argument("--output_file", default="/home/mist/projects/Wang2023/data/BCR_embed/datai/combined_distinct_light_ablang.pt",
                        type=str, help="Output file path")
    args = parser.parse_args()
    print(args)
    return args

def preprocess_sequence(sequence):
    valid_characters = "MRHKDESTNQCGPAVIFYWL*"
    return ''.join([char if char in valid_characters else '*' for char in sequence])

def sliding_window(sequence, max_length, step=10):
    if len(sequence) <= max_length:
        return [sequence]
    return [sequence[i:i + max_length] for i in range(0, len(sequence), step)]

args = parse_args()

# 璁剧疆 CUDA_LAUNCH_BLOCKING 鐜鍙橀噺
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# 鍔犺浇棰勮缁冩ā鍨?
MODEL_LOCATION = "light"  # 鍙互鏄?"heavy" 鎴?"light"
ablang_model = pretrained(MODEL_LOCATION, model_folder="/home/mist/projects/Wang2023/scripts/ablang1/model-light/", device='cuda')
ablang_model.freeze()

sequences = []
for seq_record in SeqIO.parse(args.fasta_file, "fasta"):
    preprocessed_seq = preprocess_sequence(str(seq_record.seq))
    sequences.append(preprocessed_seq)
print(f"Read {args.fasta_file} with {len(sequences)} sequences")

start_time = time.time()
batch_size = 32  # 灏濊瘯杩涗竴姝ュ噺灏戞壒澶勭悊澶у皬
max_lengths = 140  # 鏈€澶у簭鍒楅暱搴?
n_seqs = len(sequences)
dim = 768  # AbLang鐨勫祵鍏ョ淮搴?
n_batches = math.ceil(n_seqs / batch_size)
embeddings = torch.empty((n_seqs, dim), dtype=torch.float32)  # 纭繚绫诲瀷涓篺loat32

# Use the batch_loader function to iterate through batches
i = 1
for start, end, batch in batch_loader(sequences, batch_size):
    print(f'Batch {i}/{n_batches}\n')

    windowed_sequences = []
    for seq in batch:
        windowed_sequences.extend(sliding_window(seq, max_lengths))

    tokens = ablang_model.tokenizer(windowed_sequences, pad=True)
    print(f"Tokens dtype: {tokens.dtype}, device: {tokens.device}, shape: {tokens.shape}")
    print(max_lengths)
    # 寮哄埗妫€鏌ュ苟纭繚 tokens 鐨勫舰鐘朵笉瓒呰繃 max_length
    if tokens.size(1) > max_lengths:
        # raise ValueError(f"Token sequence length exceeds max_length: {tokens.size(1)} > {max_length}")
        print("token sequence length exceeds max_length: " ,tokens.size(1), ">", max_lengths)
        tokens = tokens[:, :max_lengths]
    if torch.cuda.is_available():
        tokens = tokens.to(device="cuda", dtype=torch.long, non_blocking=True)
    print(f"Tokens moved to CUDA: {tokens.dtype}, device: {tokens.device}, shape: {tokens.shape}")

    torch.cuda.empty_cache()

    with torch.no_grad():
        try:
            output = ablang_model.AbRep(tokens)
            x = output.last_hidden_states
            print(f"Output dtype: {x.dtype}, device: {x.device}, shape: {x.shape}")

            x = x.mean(dim=1)  # 璁＄畻姣忎釜瀛愬簭鍒楃殑骞冲潎琛ㄧず

            # 鑱氬悎绐楀彛鐨勭粨鏋?
            seq_embeddings = []
            current_position = 0
            for seq in batch:
                num_windows = len(sliding_window(seq, max_lengths))
                print(f"Processing sequence {seq}, num_windows: {num_windows}")

                if current_position + num_windows > x.size(0):
                    raise IndexError("Index exceeds the size of output tensor. "
                                     f"current_position: {current_position}, num_windows: {num_windows}, x.size(0): {x.size(0)}")

                seq_embedding = x[current_position:current_position + num_windows].mean(dim=0)  # 骞冲潎鑱氬悎
                seq_embeddings.append(seq_embedding)
                current_position += num_windows  # 绉婚櫎宸插鐞嗙殑绐楀彛

            embeddings[start:end] = torch.stack(seq_embeddings).cpu()

            # 棰濆璁＄畻
            x1 = torch.stack(seq_embeddings).to(torch.float32)
            x2 = nn.Linear(dim, 1).cuda()(x1)
            print(f"x2 dtype: {x2.dtype}, device: {x2.device}, shape: {x2.shape}")

            criterion_A = nn.MSELoss().cuda()  # 纭繚criterion涔熷湪cuda涓?
            max_length = max(len(string) for string in batch)
            padded_ascii_list = [torch.tensor([ord(char) / 255.0 for char in string] + [0.0] * (max_length - len(string)),
                                              dtype=torch.float32) for string in batch]
            tensor = pad_sequence(padded_ascii_list, batch_first=True).to(device="cuda", dtype=torch.float32)
            print(f"tensor dtype: {tensor.dtype}, device: {tensor.device}, shape: {tensor.shape}")

            b2 = nn.Linear(tensor.size(1), 1).cuda()(tensor)
            print(f"b2 dtype: {b2.dtype}, device: {b2.device}, shape: {b2.shape}")

            loss_A = criterion_A(b2, x2)
            print(f"Loss_A: {loss_A}, dtype: {loss_A.dtype}, device: {loss_A.device}")

        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            torch.cuda.empty_cache()
            continue

    i += 1

end_time = time.time()
print(end_time - start_time)

torch.save(embeddings, args.output_file)
