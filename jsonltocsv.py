import json
import csv


def jsonl_to_csv(jsonl_file, csv_file):
    with open(jsonl_file, 'r') as json_file, open(csv_file, 'w', newline='') as csv_output:
        fieldnames = ['seq_index', 'sequence', 'germline', 'label']
        writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        writer.writeheader()

        for line in json_file:
            data = json.loads(line)
            writer.writerow(data)

jsonl_file = r"/home/mist/EATLM/data/Bcell.germline.jsonl"
# Example usage
jsonl_to_csv(jsonl_file, "/home/mist/projects/Wang2023/data/Csv/Bcell.csv")
