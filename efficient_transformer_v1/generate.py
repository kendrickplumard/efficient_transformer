### generate model

"""
Sample from a trained model
"""
import argparse
import os
import pickle
import tiktoken
import torch

from contextlib import nullcontext

from efficient_transformer_v1.config.train_config import TRAIN_CONFIG
from efficient_transformer_v1.model import AnyModalMirasolConfig, AnyModalMirasol

# -----------------------------------------------------------------------------
train_config = TRAIN_CONFIG()
init_from = 'resume'
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="out", help="Path to the output directory")
parser.add_argument("--start", 
                    type=str, 
                    default="ANGELO:\nAnd cowards it be strawn to my bed,\nAnd thrust the gates of my threats,\nBecause he that ale away, and hang'd\nAn one with him.\n\nDUKE VINCENTIO:\nI thank your eyes against it.\n\nDUKE VINCENTIO:\nThen \
                            will answer him to save the malm:\nAnd what have you tyrannous shall do this?\n\nDUKE VINCENTIO:\nIf you have done evils of all disposition\nTo end his power, the day of thrust for a common men\nThat I leave, to fight \
                            with over-liking\nHasting in a roseman.ANGELO:\nAnd cowards it be strawn to my bed,\nAnd thrust the gates of my threats,\nBecause he that ale away, and hang'd\nAn one with him.\n\nDUKE VINCENTIO:\nI thank your eyes against \
                            it.\n\nDUKE VINCENTIO:\nThen will answer him to save the malm:\nAnd what have you tyrannous shall do this?\n\nDUKE VINCENTIO:\nIf you have done evils of all disposition\nTo end his power, the day of thrust for a common \
                            men\nThat I leave, to fight with over-liking\nHasting in a roseman.", # "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
                    help="Starting tokens for generation")
parser.add_argument("--ckpt_filename", type=str, default=f'ckpt_nl_{train_config.model_config.n_layer_enc_dec}_nh_{train_config.model_config.n_head}_nembd_{train_config.model_config.n_embd}_dr_{train_config.model_config.dropout}_wd_{train_config.weight_decay}.pt', help="Checkpoint filename")
parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum number of new tokens to generate")
parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation")
parser.add_argument("--top_k", type=int, default=8, help="Value for top-k sampling")

args = parser.parse_args()

out_dir = args.out_dir
start = args.start 
ckpt_filename = args.ckpt_filename
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens 
temperature = args.temperature
top_k = args.top_k
seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


ckpt_path = os.path.join(out_dir, ckpt_filename)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = AnyModalMirasolConfig(**checkpoint['model_args'])
print(gptconf)
model = AnyModalMirasol(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

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
print("start token:", start)
start_ids = encode(start)
print("start token length:", len(start_ids))
assert len(start_ids) > gptconf.n_groups * gptconf.latent_size[0], "minimal length in the input length"
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]) # None: add a dimension, ...: remaining dim of an array - include other dim of the original array

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')