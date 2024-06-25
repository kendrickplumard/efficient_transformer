# shakespear_char

"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import argparse
import numpy as np
import os
import pickle
import requests

from pathlib import Path

def main(data_dir: Path, train_data_proportion: float,  val_data_proportion: float):
    # download the tiny shakespeare dataset
    DATASET_PATH = Path(data_dir) / 'shakespear_char'
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    input_file_path = DATASET_PATH / 'input.txt'
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print('Downloading the dataset...')
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
    else:
        print('Found an existing dataset file, skip the downloading...')

    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n*train_data_proportion)]
    val_data = data[int(n*train_data_proportion):int(n*(train_data_proportion+val_data_proportion))]
    test_data = data[int(n*(train_data_proportion+val_data_proportion)):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    test_ids = encode(test_data)

    train_file_path = DATASET_PATH / 'train.bin'
    val_file_path = DATASET_PATH / 'val.bin'
    test_file_path = DATASET_PATH / 'test.bin'

    if not os.path.exists(train_file_path):
        print('Exporting bin files...')
        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        test_ids = np.array(test_ids, dtype=np.uint16)
        train_ids.tofile(train_file_path)
        val_ids.tofile(val_file_path)
        test_ids.tofile(test_file_path)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")
        print(f"test has {len(test_ids):,} tokens")
    else:
        print('Found an existing bin files...')
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")
        print(f"test has {len(test_ids):,} tokens")

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(DATASET_PATH / 'meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default='./data', help='data directory')
    parser.add_argument('--train_data_proportion', type=float, default=0.8, help='proportion of the train set in the overall data')
    parser.add_argument('--val_data_proportion', type=float, default=0.1, help='proportion of the val in the overall data')
    args = parser.parse_args()
    main(args.data_dir, args.train_data_proportion, args.val_data_proportion)