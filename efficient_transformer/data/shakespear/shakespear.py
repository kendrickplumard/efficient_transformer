# prepare.py

import os
import requests
import tiktoken
import numpy as np
from pathlib import Path

DATASET_PATH = Path('shakespear')
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
train_data = data[:int(n*0.8)]
val_data = data[int(n*0.8):int(n*0.9)]
test_data = data[int(n*0.9):]

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