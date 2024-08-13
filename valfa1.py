from collections import Counter

def find_duplicate_labels(fasta_file):
    labels = []
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                labels.append(line.strip())

    label_counts = Counter(labels)
    duplicates = [label for label, count in label_counts.items() if count > 1]

    return duplicates

# Example usage
fasta_file = '/home/mist/projects/Wang2023/data/FASTA/Bcell_fixed.fa'
duplicates = find_duplicate_labels(fasta_file)
if duplicates:
    print("Found duplicate sequence labels:")
    for dup in duplicates:
        print(dup)
else:
    print("No duplicate labels found.")
# # def remove_duplicate_labels(fasta_file, output_file):
# #     seen_labels = set()
# #     with open(fasta_file, 'r') as infile, open(output_file, 'w') as outfile:
# #         for line in infile:
# #             if line.startswith('>'):
# #                 label = line.strip()
# #                 if label in seen_labels:
# #                     # If a duplicate is found, modify the label to make it unique
# #                     label = f"{label}_duplicate"
# #                 seen_labels.add(label)
# #                 outfile.write(label + '\n')
# #             else:
# #                 outfile.write(line)
# #
# # # Example usage
# # fasta_file = '/home/mist/projects/Wang2023/data/FASTA/Bcell.fa'
# # output_file = '/home/mist/projects/Wang2023/data/FASTA/Bcell_fixed.fa'
# # remove_duplicate_labels(fasta_file, output_file)
# def remove_duplicate_labels(fasta_file, output_file):
#     seen_labels = set()
#     with open(fasta_file, 'r') as infile, open(output_file, 'w') as outfile:
#         write_sequence = False
#         for line in infile:


#             if line.startswith('>'):
#                 label = line.strip()
#                 if label in seen_labels:
#                     write_sequence = False  # Stop writing this sequence
#                 else:
#                     seen_labels.add(label)
#                     outfile.write(label + '\n')
#                     write_sequence = True  # Start writing this sequence
#             elif write_sequence:
#                 outfile.write(line)
#
# # 使用示例
# fasta_file = '/home/mist/projects/Wang2023/data/FASTA/Bcell.fa'
# output_file = '/home/mist/projects/Wang2023/data/FASTA/Bcell_fixed.fa'
# remove_duplicate_labels(fasta_file, output_file)
# def clean_fasta_sequences(fasta_file, output_file):
#     seen_labels = set()
#     with open(fasta_file, 'r') as infile, open(output_file, 'w') as outfile:
#         write_sequence = False
#         for line in infile:
#             if line.startswith('>'):
#                 label = line.strip()
#                 if label in seen_labels:
#                     write_sequence = False  # 停止写入这个重复序列
#                 else:
#                     seen_labels.add(label)
#                     outfile.write(label + '\n')
#                     write_sequence = True  # 开始写入这个新的序列
#             elif write_sequence:
#                 line = line.replace('*', '')  # 去除'*'字符
#                 outfile.write(line)
#
# # 使用示例
# fasta_file = '/home/mist/projects/Wang2023/data/FASTA/Bcell.fa'
# output_file = '/home/mist/projects/Wang2023/data/FASTA/Bcell_fixed.fa'
# clean_fasta_sequences(fasta_file, output_file)
