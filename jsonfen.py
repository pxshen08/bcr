import jsonlines
import random

def split_jsonl(input_file, train_output_file, val_output_file, val_ratio=0.2):
    """
    Split a JSON Lines file into train and validation sets.

    Args:
        input_file (str): Path to the input JSON Lines file.
        train_output_file (str): Path to save the output train JSON Lines file.
        val_output_file (str): Path to save the output validation JSON Lines file.
        val_ratio (float): Ratio of validation data. Default is 0.2 (20%).
    """
    with jsonlines.open(input_file, 'r') as reader:
        data = list(reader)

    random.shuffle(data)

    split_index = int(len(data) * (1 - val_ratio))

    train_data = data[:split_index]
    val_data = data[split_index:]

    with jsonlines.open(train_output_file, 'w') as writer:
        writer.write_all(train_data)

    with jsonlines.open(val_output_file, 'w') as writer:
        writer.write_all(val_data)

# Example usage:
split_jsonl("/home/mist/EATLM/data/Sars.germline.jsonl", "/home/mist/EATLM/data/Sarstrain.jsonl", "/home/mist/EATLM/data/Sarsval.jsonl")
