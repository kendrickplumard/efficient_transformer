### generate model

"""
Sample from a trained model
"""
import argparse
import os
import pickle
import schedulefree
import tiktoken
import torch

from contextlib import nullcontext

from vanilla_transformer.model import GPT, GPTConfig
from vanilla_transformer.train_utils import TRAIN_CONFIG

# -----------------------------------------------------------------------------
# Initialize argument parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--init_from", type=str, default='resume', help="Initialization method")
parser.add_argument("--out_dir", type=str, default="out", help="Path to the output directory")
parser.add_argument("--ckpt_filename", type=str, default='gpt.pt', help="Checkpoint filename")
parser.add_argument("--start", 
                    type=str, 
                    default="\n", # "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
                    help="Starting tokens for generation")
parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum number of new tokens to generate")
parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation")
parser.add_argument("--top_k", type=int, default=8, help="Value for top-k sampling")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use (e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:1')")
parser.add_argument("--dtype", type=str, default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16', help="Data type ('float32', 'bfloat16', or 'float16')")
parser.add_argument("--compile", action='store_true', help="Use PyTorch 2.0 to compile the model (optional)")

# Parse arguments
args = parser.parse_args()

# Update variables with parsed arguments
init_from = args.init_from
out_dir = args.out_dir
ckpt_filename = args.ckpt_filename
start = args.start
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens
temperature = args.temperature
top_k = args.top_k
seed = args.seed
device = args.device
dtype = args.dtype
compile = args.compile
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
train_config = TRAIN_CONFIG()
device_type = 'cuda' if 'cuda' in device else 'cpu' 
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
ckpt_path = os.path.join(out_dir, ckpt_filename)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args']) # Assuming GPTConfig is defined
model = GPT(gptconf) # Assuming GPT is defined
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
if train_config.use_schedule_free_lr:
    optimizer = schedulefree.AdamWScheduleFree(model.parameters()) # they update warmup counter every optimizer step
    optimizer.load_state_dict(checkpoint['optimizer'])

model.eval()
model.to(device)
if train_config.use_schedule_free_lr:
  optimizer.eval()
if compile:
    model = torch.compile(model) 

# look for the meta pickle in case it is available in the dataset folder
load_meta = True
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join(checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]) # None: add a dimension, ...: remaining dim of an array - include other dim of the original array

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
