# prepare.py

import argparse  
import numpy as np
import os
import requests
import tiktoken

from pathlib import Path

DATASET_PATH = Path('data/shakespear_char')
DATASET_PATH.mkdir(parents=True, exist_ok=True)

def main(data_dir: Path, train_data_proportion: float, val_data_proportion: float):
    global DATASET_PATH
    DATASET_PATH = Path(data_dir) / 'shakespear_char'
    DATASET_PATH.mkdir(parents=True, exist_ok=True)

    # download the tiny shakespeare dataset
    input_file_path = DATASET_PATH / 'input.txt'
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print('Downloading the dataset')
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
    else:
        print('Found an existing dataset file, skip the downloading...')

    with open(input_file_path, 'r') as f:
        data = f.read()
    n = len(data)

    # Utiliser les proportions pour diviser les données
    train_data = data[:int(n * train_data_proportion)]
    val_data = data[int(n * train_data_proportion):int(n * (train_data_proportion + val_data_proportion))]
    test_data = data[int(n * (train_data_proportion + val_data_proportion)):]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    test_ids = enc.encode_ordinary(test_data)

    train_file_path = DATASET_PATH / 'train.bin'
    val_file_path = DATASET_PATH / 'val.bin'
    test_file_path = DATASET_PATH / 'test.bin'

    if not os.path.exists(train_file_path):
        # export to bin files
        print('Exporting bin files...')
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
        print('Found existing bin files...')
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")
        print(f"test has {len(test_ids):,} tokens")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default='./data', help='data directory')
    parser.add_argument('--train_data_proportion', type=float, default=0.8, 
                        help='proportion of the train set in the overall data')
    parser.add_argument('--val_data_proportion', type=float, default=0.1, 
                        help='proportion of the val in the overall data')
    args = parser.parse_args()
    main(args.data_dir, args.train_data_proportion, args.val_data_proportion)