import csv

def csv_to_fasta(csv_file, fasta_file):
    with open(csv_file, 'r') as csvfile, open(fasta_file, 'w') as fastafile:
        # reader = csv.reader(csvfile,header=None)
        reader = csv.reader(csvfile)
        for row in reader:
            identifier, sequence = row
            fastafile.write(f'>{identifier}\n{sequence}\n')

csv_to_fasta("/home/mist/BCR-SORT-master/data/Bcell_1f.csv", '/home/mist/BCR-SORT-master/data/Bcell_1f.fasta')

