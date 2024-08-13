
import csv

def csv_to_annotation(csv_file, output_file):
    with open(csv_file, 'r') as csvfile, open(output_file, 'w') as outfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            annotation_line = " ".join(row) + "\n"
            outfile.write(annotation_line)

# 调用函数进行转换
csv_file = r'/home/mist/projects/Wang2023/data/Csv/Bcell.csv'
output_file = r'/home/mist/projects/Wang2023/data/Annotations/Bcell.txt'
csv_to_annotation(csv_file, output_file)
