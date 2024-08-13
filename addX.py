from Bio import SeqIO

# 读取序列并找到最长的序列长度
sequences = list(SeqIO.parse("/home/mist/BCR-SORT-master/data/Bcell_1f.fasta", "fasta"))
max_length = max(len(seq.seq) for seq in sequences)

# 在较短的序列前面添加字符“X”
for seq in sequences:
    seq.seq = "X" * (max_length - len(seq.seq)) + str(seq.seq)

# 将修改后的序列写入新的FASTA文件
with open("/home/mist/BCR-SORT-master/data/Bcell_1f.fasta", "w") as output_handle:
    SeqIO.write(sequences, output_handle, "fasta")

print("序列已填充并写入新的FASTA文件。")
