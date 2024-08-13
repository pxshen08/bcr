from Bio import SeqIO

input_fastq = "path/to/your/file.fastq"
output_fasta = "path/to/your/output.fasta"

# Convert FASTQ to FASTA
count = SeqIO.convert(input_fastq, "fastq", output_fasta, "fasta")
print(f"Converted {count} records")
