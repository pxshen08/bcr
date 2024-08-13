import pandas as pd
import os
import random

# # 读取原始CSV文件
# df = pd.read_csv(r'/home/mist/projects/Wang2023/data/Csv/cdr3cn0729.csv')
#
# df = df.sample(frac=1).reset_index(drop=True)
# # 计算每个小文件应该包含的行数
# subset_size = 1+len(df) // 100
#
# # 按顺序将数据分成500个小的子集
# subsets = [df.iloc[i:i+subset_size] for i in range(0, len(df), subset_size)]
#
# # 将每个子集的数据写入一个独立的CSV文件
# output_folder = r'/home/mist/projects/Wang2023/data/Csv/'
# os.makedirs(output_folder, exist_ok=True)
#
# for i, subset in enumerate(subsets):
#     subset.to_csv(os.path.join(output_folder, f"cdr3_{i+1}.csv"), index=False)
# from Bio import SeqIO
# import random
# import os
#
# # 读取原始FASTA文件
# input_fasta = r'/home/mist/projects/Wang2023/data/FASTA/cdr3seq1.fasta'
#
# # 读取FASTA文件中的所有记录
# records = list(SeqIO.parse(input_fasta, "fasta"))
#
# # 随机打乱记录的顺序
# random.shuffle(records)
#
# # 计算每个小文件应该包含的序列数
# subset_size = 1 + len(records) // 200
#
# # 按顺序将数据分成500个小的子集
# subsets = [records[i:i + subset_size] for i in range(0, len(records), subset_size)]
#
# # 将每个子集的数据写入一个独立的FASTA文件
# output_folder = r'/home/mist/ClonalTree/Data/Real_dataset/'
# os.makedirs(output_folder, exist_ok=True)
#
# for i, subset in enumerate(subsets):
#     output_fasta = os.path.join(output_folder, f"cdr3seq1_{i + 1}.fasta")
#     with open(output_fasta, "w") as output_handle:
#         SeqIO.write(subset, output_handle, "fasta")
#
# print(f"成功将 {input_fasta} 随机打乱并分割成 {len(subsets)} 个较小的FASTA文件，保存在 {output_folder} 文件夹中。")

from Bio import SeqIO
import os

# 定义文件路径
small_fasta = r'/home/mist/ClonalTree/Data/Real_dataset/cdr3aa1_1.fasta'
large_fasta = r'/home/mist/projects/Wang2023/data/FASTA/cdr3seq1.fasta'
output_fasta = r'/home/mist/ClonalTree/Data/Real_dataset/cdr3seq1_1.fasta'

# 读取小的FASTA文件，获取索引号列表
small_indices = [record.id for record in SeqIO.parse(small_fasta, "fasta")]

# 读取大的FASTA文件，根据索引号列表查找对应的序列
matched_records = []
for record in SeqIO.parse(large_fasta, "fasta"):
    if record.id in small_indices:
        matched_records.append(record)

# 将找到的序列保存到新的FASTA文件中
with open(output_fasta, "w") as output_handle:
    SeqIO.write(matched_records, output_handle, "fasta")

print(f"成功将匹配的序列保存到 {output_fasta}")
