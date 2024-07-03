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

from efficient_transformer_v2.model import ConditionalTF, ConditionalTFConfig
from efficient_transformer_v2.train_utils import TRAIN_CONFIG

# -----------------------------------------------------------------------------
train_config = TRAIN_CONFIG() # use the default train config
init_from = 'resume'
out_dir = 'out'
ckpt_filename = 'conditional_tf.pt'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 8 # retain only the top_k most likely tokens, clamp others to have 0 probability
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

# model
# ckpt_path = list(glob.glob(os.path.join(out_dir, '*.pt')))[0]
ckpt_path = os.path.join(out_dir, ckpt_filename)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = ConditionalTFConfig(**checkpoint['model_args'])
model = ConditionalTF(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
if train_config.use_schedule_free_lr:
  optimizer = schedulefree.AdamWScheduleFree([p for pn, p in model.named_parameters() if 'sampling_predictor' not in pn]) 
  optimizer_sampling_predictor = schedulefree.AdamWScheduleFree([p for pn, p in model.named_parameters() if 'sampling_predictor' in pn]) 
  optimizer.load_state_dict(checkpoint['optimizer'])
  optimizer_sampling_predictor.load_state_dict(checkpoint['optimizer_sampling_predictor'])

model.eval()
model.to(device)
if train_config.use_schedule_free_lr:
  optimizer.eval()
  optimizer_sampling_predictor.eval()
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
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
