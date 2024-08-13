# import json
#
# def jsonl_to_fasta(jsonl_file, fasta_file):
#     with open(jsonl_file, 'r') as json_file, open(fasta_file, 'w') as fasta:
#         for line in json_file:
#             data = json.loads(line)
#             sequence_id = data.get("seq_index", "")
#             sequence = data.get("sequence", "")
#             if sequence_id and sequence:
#                 fasta.write(f">{sequence_id}\n{sequence}\n")
#
# jsonl_file = r"/home/mist/EATLM/data/Bcell.germline.jsonl"
# # Example usage
# jsonl_to_fasta(jsonl_file, "/home/mist/projects/Wang2023/data/FASTA/Bcell.fasta")
def find_longest_sequence(sequences):
    return max(sequences, key=len)

# def fill_missing_characters(sequences, longest_sequence):
#     filled_sequences = []
#     max_length = len(longest_sequence)
#     for seq in sequences:
#         if len(seq) < max_length:
#             diff = max_length - len(seq)
#             start_fill = diff // 2
#             end_fill = diff - start_fill
#             filled_seq = "-" * start_fill + seq + "-" * end_fill
#             filled_sequences.append(filled_seq)
#         else:
#             filled_sequences.append(seq)
#     return filled_sequences
def fill_missing_characters(sequences, longest_sequence):
    filled_sequences = []
    max_length = len(longest_sequence)
    for seq in sequences:
        if len(seq) < max_length:
            diff = max_length - len(seq)
            start_fill = diff // 2
            end_fill = diff - start_fill
            filled_seq = seq[:start_fill] + "-" * diff + seq[start_fill:]
            filled_sequences.append(filled_seq)
        else:
            filled_sequences.append(seq)
    return filled_sequences

# def split_fasta(input_fasta, output_prefix, num_files):
#     sequences = []
#     with open(input_fasta, 'r') as f:
#         current_sequence = ""
#         for line in f:
#             if line.startswith('>'):
#                 if current_sequence:
#                     sequences.append(current_sequence)
#                 current_sequence = ""
#             else:
#                 current_sequence += line.strip()
#         sequences.append(current_sequence)
#
#     # Find the longest sequence
#     longest_sequence = find_longest_sequence(sequences)
#
#     # Fill missing characters in sequences
#     filled_sequences = fill_missing_characters(sequences, longest_sequence)
#
#     # Split and write to files
#     split_fasta_files(output_prefix, num_files, filled_sequences)
#
# def split_fasta_files(output_prefix, num_files, sequences):
#     sequences_per_file = len(sequences) // num_files
#     remainder = len(sequences) % num_files
#
#     current_file_index = 1
#     current_sequences = []
#
#     for seq in sequences:
#         current_sequences.append(seq)
#
#         if len(current_sequences) == sequences_per_file + (1 if remainder > 0 else 0):
#             write_sequences_to_file(f"{output_prefix}_{current_file_index}.fasta", current_sequences)
#             current_file_index += 1
#             current_sequences = []
#             remainder -= 1
#
#     # Write remaining sequences to the last file
#     if current_sequences:
#         write_sequences_to_file(f"{output_prefix}_{current_file_index}.fasta", current_sequences)
#
#
# def write_sequences_to_file(file_name, sequences_with_id):
#     with open(file_name, 'w') as f:
#         for seq_id, seq in sequences_with_id:
#             f.write(f">{seq_id}\n{seq}\n")
def read_fasta(input_fasta):
    sequences_with_id = []
    with open(input_fasta, 'r') as f:
        current_id = None
        current_sequence = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences_with_id.append((current_id, current_sequence))
                current_id = line[1:]
                current_sequence = ""
            else:
                current_sequence += line
        # Add the last sequence
        if current_id is not None:
            sequences_with_id.append((current_id, current_sequence))
    return sequences_with_id
def split_fasta(input_fasta, output_prefix, num_files):
    sequences_with_id = read_fasta(input_fasta)

    # Extract sequences from (id, sequence) tuples
    sequences = [sequence for _, sequence in sequences_with_id]

    # Find the longest sequence
    longest_sequence = find_longest_sequence(sequences)

    # Fill missing characters in sequences
    filled_sequences = fill_missing_characters(sequences, longest_sequence)

    # Split and write to files
    split_fasta_files(output_prefix, num_files, sequences_with_id, filled_sequences)

def split_fasta_files(output_prefix, num_files, sequences_with_id, sequences):
    sequences_per_file = len(sequences) // num_files
    remainder = len(sequences) % num_files

    current_file_index = 1
    current_sequences = []
    current_sequences_with_id = []

    for (seq_id, seq), filled_seq in zip(sequences_with_id, sequences):
        current_sequences_with_id.append((seq_id, filled_seq))
        current_sequences.append(filled_seq)
        if len(current_sequences) == sequences_per_file + (1 if remainder > 0 else 0):
            write_sequences_to_file(f"{output_prefix}_{current_file_index}.fasta", current_sequences_with_id, current_sequences)
            current_file_index += 1
            current_sequences = []
            current_sequences_with_id = []
            remainder -= 1

    # Write remaining sequences to the last file
    if current_sequences:
        write_sequences_to_file(f"{output_prefix}_{current_file_index}.fasta", current_sequences_with_id, current_sequences)

def write_sequences_to_file(file_name, sequences_with_id, sequences):
    with open(file_name, 'w') as f:
        for (seq_id, _), seq in zip(sequences_with_id, sequences):
            f.write(f">seq{seq_id}\n{seq}\n")

# 在之后的代码保持不变


# Example usage
split_fasta("/home/mist/projects/Wang2023/data/FASTA/Bcell.fasta", "/home/mist/projects/Wang2023/data/FASTA1/Bcell", 500)
