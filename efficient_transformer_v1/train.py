import argparse
import atexit
import copy
import numpy as np
import os
import pickle
import psutil
import time
import torch

from contextlib import nullcontext
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from efficient_transformer_v1.train_utils import *

torch._dynamo.config.suppress_errors = True
torch.autograd.set_detect_anomaly(True)

main_timer = time.time()

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_filename', type=str, default=f'ckpt_nl_{TRAIN_CONFIG().model_config.n_layer_enc_dec}_nh_{TRAIN_CONFIG().model_config.n_head}_nembd_{TRAIN_CONFIG().model_config.n_embd}_dr_{TRAIN_CONFIG().model_config.dropout}_wd_{TRAIN_CONFIG().weight_decay}.pt', 
                    help='Name of the checkpoint file')
args = parser.parse_args()

train_config = TRAIN_CONFIG()
tokens_per_iter = train_config.gradient_accumulation_steps * train_config.maxBS * train_config.model_config.block_size
print(f"tokens per iteration will be at most: {tokens_per_iter:,}")


os.makedirs(train_config.out_dir, exist_ok=True)
ckpt_filename = args.ckpt_filename  # Use the provided checkpoint filename
config = copy.deepcopy(vars(train_config))
config['model_config'] = copy.deepcopy(vars(config['model_config'])) # get a dict from the AnyModalMirasolConfig instance
torch.manual_seed(train_config.seed)
torch.cuda.manual_seed(train_config.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
writer = SummaryWriter(log_dir = os.path.join(train_config.log_dir, ckpt_filename)) # tensorboard init



train_dataset = CustomDataset(os.path.join(train_config.dataset, 'train.bin'), train_config.model_config.block_size)
val_dataset = CustomDataset(os.path.join(train_config.dataset, 'val.bin'),
                            train_config.model_config.block_size, train_config.restrict_val_iters,
                            train_config.step_size_eval)
train_dataloader = DataLoader(dataset = train_dataset,
                              batch_size = train_config.batch_size,
                              num_workers = os.cpu_count(),
                              shuffle = True,
                              pin_memory = True,
                              drop_last = True)
val_dataloader = DataLoader(dataset = val_dataset,
                            batch_size = train_config.val_batch_size,
                            num_workers = os.cpu_count(),
                            shuffle = False,
                            pin_memory = True,
                            drop_last = True)
print(f"number of batch in the train set: {len(train_dataloader)}")
print(f"number of params's update per epoch: {len(train_dataloader) // train_config.gradient_accumulation_steps}")

# attempt to derive vocab_size from the dataset
DATASET_PATH = Path(train_config.dataset)
meta_path = DATASET_PATH / 'meta.pkl'
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")


# init these up here, can override if init_from='resume' (i.e. from a checkpoint) # peut etre creer une dataclass pour ces args, ckptMetaData
model, model_args, checkpoint, iter, best_val_loss = buildModel(train_config.init_from, device, meta_vocab_size, train_config.out_dir, copy.deepcopy(vars(train_config.model_config)), ckpt_filename)
if train_config.model_config.block_size < model.config.block_size:
    model.crop_block_size(train_config.model_config.block_size)
    model_args['block_size'] = train_config.model_config.block_size # so that the checkpoint will have the right value
model.to(device)

# init model stats
if train_config.register_stats_wb:
  activation_stats_history, weight_stats_history, activation_stats = init_model_stats(model, (train_dataset[0][0]).expand(train_config.batch_size, -1))


# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(train_config.weight_decay, train_config.learning_rate, (train_config.beta1, train_config.beta2), device)
if train_config.init_from == 'resume':
  optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if train_config.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# training loop
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
start_epoch = iter // len(train_dataloader)
max_iters = len(train_dataloader) * train_config.num_epochs

# cleaning
def cleaning():
  train_dataloader = None
  val_dataloader = None
  print("EXECUTION TIME:", time.time() - t0)
atexit.register(cleaning)

for epoch in range(start_epoch, train_config.num_epochs):

  # forward backward update, with optional gradient accumulation to simulate larger batch size
  # and using the GradScaler if data type is float16
  model.train()
  total_loss = 0.0
  total_length = len(train_dataloader) // train_config.gradient_accumulation_steps + int (len(train_dataloader) % train_config.gradient_accumulation_steps != 0)
  pbar = tqdm(colour="blue", desc=f"Training epoch: {epoch + 1}", total = total_length, dynamic_ncols = True)
  for step, (X, y) in enumerate(train_dataloader): # train_dataloader
    # determine and set the learning rate for this iteration
    lr = calc_lr(train_config, (epoch + 1) * (step + 1), max_iters)
    if model_args['muP']:
      for param_group in optimizer.param_groups: # 2 param_groups when i set the lr's value ca it overrides the default one
        param_group['lr'] = lr / param_group['muPScale']

    coeff_linear_warmup_latent_causal_loss = 1
    max_step_warmup_latent_loss = train_config.latent_causal_loss_warmup_max_step * max_iters
    if (model_args['latent_causal_loss_type'] != None) and (epoch == 0) and (((epoch + 1) * (step + 1)) < max_step_warmup_latent_loss):
      coeff_linear_warmup_latent_causal_loss = ((epoch + 1) * (step + 1)) / max_step_warmup_latent_loss

    X, y = X.to(device), y.to(device)
    with ctx:
      logits, loss = model(X, y, coeff_linear_warmup_latent_causal_loss * train_config.latent_causal_modeling_loss_coeff, train_config.step_size_eval)
      loss = {key: loss[key] / train_config.gradient_accumulation_steps for key in loss} # scale the loss to account for gradient accumulation


    total_loss += loss["total"].detach().float()
    scaler.scale(loss["total"]).backward()
    if ((step + 1) % train_config.gradient_accumulation_steps == 0) or (step == len(train_dataloader) -1):
      # clip the gradient
      if train_config.grad_clip != 0.0:
        scaler.unscale_(optimizer) # Each parameter’s gradient (.grad attribute) should be unscaled before the optimizer updates the parameters, so the scale factor does not interfere with the learning rate.
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
      # step the optimizer and scaler if training in fp16
      scaler.step(optimizer) # As part of the unscale_(), gradients are checked for infs/NaNs; If no inf/NaN gradients are found, invokes optimizer.step() using the unscaled gradients. Otherwise, optimizer.step() is skipped to avoid corrupting the params.
      scaler.update()

      if (step + 1) % train_config.eval_interval == 0: # deplacer ici m'a rajouté de la ram
        val_loss = estimate_loss(model, val_dataloader, ctx, device)
        if val_loss < best_val_loss or train_config.always_save_checkpoint:
          best_val_loss = val_loss
          mem = psutil.virtual_memory()
          print(f"\nMemoire disponible/Memoire totale: {mem.available//1e9}/{mem.total//1e9}")
          train_loss = {key: loss[key].detach().float() * train_config.gradient_accumulation_steps for key in loss}
          print(f"\nStart saving ckpt at step {step + 1}, epoch {epoch + 1} "
                f"with total train loss {train_loss['total']:.4f}, \n"
                f"cross entropy train loss {train_loss['cross_entropy']:.4f}, "
                f"latent_causal_modeling train loss {train_loss['latent_causal_modeling']:.4f}, \n"
                f"and val loss {val_loss:.4f}")
          saveCkpt(val_loss, model, optimizer, model_args, (step + 1) * (epoch + 1), config, os.path.join(train_config.out_dir, ckpt_filename))
        # writeLogs(model, X, writer, {'train': loss.detach().float() * train_config.gradient_accumulation_steps, 'val': val_loss}, (step + 1) * (epoch + 1), lr)

      # if (step + 1) % train_config.nb_grad_update_to_register_stats and train_config.register_stats_wb:
      #   weight_stats_history.append(get_weight_stats(model))
      #   for name, stats in activation_stats.items():
      #     activation_stats_history[name].append(np.mean(stats))


      # flush the gradients as soon as we can, no need for this memory anymore
      optimizer.zero_grad(set_to_none=True)
      # pbar.refresh()
      pbar.update(1)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    local_iter_num += 1
 
  pbar.close()
train_perplexity = torch.exp(total_loss / len(train_dataloader))

writer.close()
val_dataloader = None
train_dataloader = None
print("\ntime elapsed: ", (time.time() - main_timer) / 3600)