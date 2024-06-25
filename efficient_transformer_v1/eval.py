### test model

import argparse
import numpy as np
import os
import torch

from contextlib import nullcontext

from torch.utils.data import DataLoader

from efficient_transformer_v1.config.train_config import TRAIN_CONFIG
from efficient_transformer_v1.model import AnyModalMirasol, AnyModalMirasolConfig
from efficient_transformer_v1.train_utils import CustomDataset, estimate_loss

torch._dynamo.config.suppress_errors = True

train_config = TRAIN_CONFIG()

parser = argparse.ArgumentParser()
# Add argument for checkpoint filename
parser.add_argument("--ckpt_filename", type=str, 
                    help="Name of the checkpoint file to load.", 
                    default=f'ckpt_nl_{train_config.model_config.n_layer_enc_dec}_nh_{train_config.model_config.n_head}_nembd_{train_config.model_config.n_embd}_dr_{train_config.model_config.dropout}_wd_{train_config.weight_decay}.pt')
# Parse the arguments
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Access the checkpoint filename from args
ckpt_filename = args.ckpt_filename 
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
torch.manual_seed(train_config.seed)
torch.cuda.manual_seed(train_config.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# -----------------------------------------------------------------------------

# model
ckpt_path = os.path.join(train_config.out_dir, ckpt_filename)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = AnyModalMirasolConfig(**checkpoint['model_args'])
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

# load dataset
test_dataset = CustomDataset(os.path.join(train_config.dataset, 'test.bin'),
                             train_config.model_config.block_size, train_config.restrict_eval_iters,
                             train_config.step_size_eval)
test_dataloader = DataLoader(dataset = test_dataset,
                              batch_size = 2 * train_config.batch_size, # could allow double the train bs, keep no grad in eval mode
                              num_workers = os.cpu_count(),
                              shuffle = True,
                              pin_memory = True)


print(f"Here is the test loss of the model: {estimate_loss(model, test_dataloader, ctx, device)}")