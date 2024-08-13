from Bio.Blast.Applications import NcbiblastnCommandline
fasta_file="/home/mist/projects/Wang2023/data/FASTA/combined_cdr3_heavy.fa"
# 设置 IgBlast 的命令行参数
igblastn_cline = NcbiblastnCommandline(query=fasta_file, db="path/to/igblast_db", outfmt=7, out="igblast_results.txt")

# 运行 IgBlast
stdout, stderr = igblastn_cline()
print("IgBlast 比对完成。结果保存在 igblast_results.txt 中。")
