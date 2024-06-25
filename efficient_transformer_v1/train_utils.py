import math
import numpy as np
import torch

from contextlib import nullcontext
from functools import partial
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Union

from efficient_transformer_v1.config.train_config import TRAIN_CONFIG
from efficient_transformer_v1.model import AnyModalMirasol, AnyModalMirasolConfig

@torch.no_grad()
def writeLogs(model: torch.nn.Module, input: torch.Tensor, writer: SummaryWriter, losses: Dict[str, float], iter: int, lr: float) -> None:
  model.eval()
  writer.add_scalars(main_tag = "Loss", tag_scalar_dict = {"train_loss": losses['train'], "val_loss": losses['val']}, global_step=iter)
  writer.add_scalars(main_tag = "Learning rate", tag_scalar_dict = {"lr": lr}, global_step = iter)
  for name, p in model.named_parameters():
    if p.dim() >=2: # skip bias, gamma, beta from the layer norm for simplicity
      writer.add_histogram(tag = name, values = p, global_step = iter)
      if (p.grad == None): print("no grad", name)
      writer.add_histogram(tag = name + '/grad', values = p.grad, global_step = iter)
      # writer.add_histogram(tag = name + '/lr_grad_over_p', values = ((lr*p.grad).std() / (p.std())).log10(), global_step = iter) # strange internal bugs pytorchnwith muP reports ln weights, without doesn't report

  # for name, layer in model.named_modules():
  #   if isinstance(layer, torch.nn.GELU):
  #     writer.add_histogram(tag = name + '/non_linearity', values = p, global_step = iter)
  #     writer.add_histogram(tag = name + '/non_linearity_grad', values = p.grad, global_step = iter)
  #   else:
  #     writer.add_histogram(tag = name, values = p, global_step = iter)
  #     writer.add_histogram(tag = name + '/grad', values = p.grad, global_step = iter)

  # writer.add_graph(model = model, input_to_model = input)
  model.train()

@torch.no_grad()
def saveCkpt(best_val_loss: float, model: torch.nn.Module, optimizer: torch.optim, model_args: Dict, iter: int, config: Dict, filepath: str) -> float:
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter,
        'best_val_loss': best_val_loss,
        'config': config,
        }
    file_path = filepath
    torch.save(
        checkpoint,
        file_path
        )
    print(f"\nSaved to {file_path}")

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader, ctx: Union[torch.amp.autocast, nullcontext], device: str) -> float:
    model.eval()
    losses = torch.zeros(len(val_dataloader))
    for step, (X, y) in enumerate(val_dataloader):
      X, Y = X.to(device), y.to(device)
      with ctx: # In these regions, ops run in an op-specific dtype chosen by autocast # Backward passes under autocast are not recommended. Backward ops run in the same type that autocast used for corresponding forward ops.
        # print(X[0])
        logits, loss = model(X, Y) # va falloir degager loss de forward pass
        losses[step] = loss["cross_entropy"].item()
    model.train()
    return losses.mean()

def buildModel(init_from: str, device: str, meta_vocab_size: int, out_dir: str, default_model_args: Dict, ckpt_filename: str = None):
  if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    default_model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    model_args = default_model_args
    gptconf = AnyModalMirasolConfig(**model_args)
    model = AnyModalMirasol(gptconf)
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter = 0
    best_val_loss = 1e9
    checkpoint = None
  elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, ckpt_filename)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    model_args = default_model_args
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer_enc_dec', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = AnyModalMirasolConfig(**model_args)
    model = AnyModalMirasol(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
  return model, model_args, checkpoint, iter, best_val_loss

def linear_schedule_bs(current_bs: int, current_step: int, start_step: int, max_bs: int, coeff: int, icr_interval: int):
  mult_bs = coeff * current_bs
  if (current_step <= start_step) or (mult_bs >= max_bs):
    return current_bs
  elif (current_step - start_step) % icr_interval == 0:
    return mult_bs
  else:
    return current_bs


# learning rate decay scheduler (cosine with warmup)
def cos_schedule_lr(current_step: int, train_config: TRAIN_CONFIG, max_iters: int) -> float:
  # 1) linear warmup for warmup_iters steps
  if current_step < (train_config.warmup_iters * max_iters):
    return train_config.learning_rate * current_step / (train_config.warmup_iters * max_iters)
  # 2) if current_step > lr_decay_iters, return min learning rate
  if current_step > (train_config.lr_decay_iters * max_iters):
    return train_config.min_lr
  # 3) in between, use cosine decay down to min learning rate # count steps from the initiation of the cos decay
  decay_ratio = (current_step - train_config.warmup_iters * max_iters) / (train_config.lr_decay_iters * max_iters - train_config.warmup_iters * max_iters)
  assert 0.0 <= decay_ratio <= 1.0, "something wrong with the cosine schedule config"
  coeff = (1/2) * (1.0 + math.cos(math.pi * decay_ratio)) 
  lr = train_config.learning_rate + coeff * (train_config.min_lr - train_config.learning_rate)


# learning rate decay scheduler (linear with warmup)
def linear_schedule_lr(current_step: int, train_config: TRAIN_CONFIG, max_iters: int):
  # 1) linear warmup for warmup_iters steps
  if current_step < (train_config.warmup_iters * max_iters):
    return train_config.learning_rate * current_step / (train_config.warmup_iters * max_iters)
  # 2) if current_step > lr_decay_iters, return min learning rate
  if current_step > (train_config.lr_decay_iters * max_iters):
    return train_config.min_lr
  # 3) in between, use linear decay down to min learning rate
  coeff = (current_step - train_config.warmup_iters * max_iters) / (train_config.lr_decay_iters * max_iters - train_config.warmup_iters * max_iters) # starts with 0 and ends with 1
  return train_config.learning_rate + coeff * (train_config.min_lr - train_config.learning_rate)

def calc_lr(train_config: TRAIN_CONFIG, current_step: int, max_iters: int):
  if train_config.lr_schedule == 'constant':
    return train_config.learning_rate
  elif train_config.lr_schedule == 'cos_schedule':
    return cos_schedule_lr(current_step, train_config, max_iters)
  elif train_config.lr_schedule == 'linear_schedule':
    return linear_schedule_lr(current_step, train_config, max_iters)
  else:
    raise ValueError("You must set a lr schedule between constant/cos_schedule/linear_schedule")


def init_model_stats(model, sample):
  activation_stats = {}

  # register activations means
  def save_activation_stats(name, mod, inp, out):
    activation_stats[name].append(out.norm(2).mean().item())

  # attach activation hook
  for name, module in model.named_modules():
  # print("name modules", name)
    if isinstance(module, nn.Linear):
      print("name linear modules", name)
      activation_stats[name] = []
      module.register_forward_hook(partial(save_activation_stats, name))

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model(sample.to(device))
  activation_stats_history = {name: [] for name in activation_stats.keys()}
  weight_stats_history = []
  return activation_stats_history, weight_stats_history, activation_stats

# register means weights per layer
def get_weight_stats(model):
  weight_stats = {}
  for name, param in model.named_parameters():
    if "weight" in name:
      weight_stats[name] = param.data.norm(2).mean().item()
  return weight_stats

# Note: We're using the pin_memory=True parameter in the create_dataloaders() function to speed up computation. pin_memory=True avoids unnecessary
# copying of memory between the CPU and GPU memory by "pinning" examples that have been seen before. Though the benefits of this will likely be seen
# with larger dataset sizes

def buildDataBlock(train_data: np.ndarray, block_size: int, limitSizeTo: int = None, stepSize: int = 1):
  maxElm = (min(limitSizeTo, len(train_data) - block_size)) if limitSizeTo else  (len(train_data) - block_size)
  X = [torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in range(0, maxElm, stepSize)] # rajouter un char end of sentence plutot
  y = [torch.from_numpy((train_data[i+1:i+1+block_size]).astype(np.int64)) for i in range(0, maxElm, stepSize)]
  return X, y

class CustomDataset(Dataset):
  def __init__(self, target_file: str, block_size: int, limitSizeTo: int = None, stepSize: int = 1) -> None:
    self.train_data = np.memmap(target_file, dtype = np.uint16, mode = 'r')
    self.block_size = block_size
    self.data_block = buildDataBlock(self.train_data, self.block_size, limitSizeTo, stepSize)

  def __len__(self) -> int:
    return len(self.data_block[0])

  def __getitem__(self, index: int):
    X, y = self.data_block[0], self.data_block[1]
    return X[index], y[index]

