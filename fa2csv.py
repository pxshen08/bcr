import pandas as pd
# 载入数据
from Bio import SeqIO

meta = []
sequence = []
seq = ('/home/mist/projects/Wang2023/data/FASTA/Bcell_1f.fasta')
for seq_record in SeqIO.parse(seq, "fasta"):
    meta.append(str(seq_record.id))
    sequence.append(str(seq_record.seq))
#print(sequence)
df= pd.DataFrame(data ={'Meta':meta,'SequenceID':sequence})
print(df)

# 数据存入csv

df.to_csv("/home/mist/BCR-SORT-master/data/cdr3seq1.csv", sep=',', index=False)