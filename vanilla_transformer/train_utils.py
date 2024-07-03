import math
import numpy as np
import os
import torch
import wandb

from contextlib import nullcontext
from torch.utils.data import Dataset
from typing import Dict, Union, Optional

from vanilla_transformer.config.train_config import TRAIN_CONFIG
from vanilla_transformer.model import GPT, GPTConfig

@torch.no_grad()
def writeLogs(model: torch.nn.Module, 
              activations: Dict[str, torch.Tensor], 
              losses: Dict, iter: int, lr: float,  
              optimizer: torch.optim.Optimizer, 
              train_config: TRAIN_CONFIG, 
              model_args: Dict) -> None:
  
  wandb.log({
      'iter': iter,
      'train_loss': losses['train_loss'],
      'val_loss': losses['val_loss'],
      })
  
  for module, activation in activations.items():
    wandb.log({
        f'activations/{module}_mean': activation.mean().item(),
        f'activations/{module}_std': activation.std().log10().item(),
        f'activations/{module}_norm2': activation.norm(2).mean().log10().item()
    })

  if train_config.use_schedule_free_lr:
    k = optimizer.param_groups[0]['k']
    warmup_steps = optimizer.param_groups[0]['warmup_steps']
    sched = (k + 1) / warmup_steps if (k < warmup_steps) else 1.0
    lr = optimizer.param_groups[0]['lr']*sched*math.sqrt(1-optimizer.param_groups[0]['betas'][1]**(k+1))

  special_weights = ['pos_embd_layers', 'wte', 'first_group_tokens_for_upsampling', 'shared_soft_experts', 'mem_token']
  for pn, p in model.named_parameters():
    wandb.log({
        f'params/{pn}_mean': p.data.mean().item(),
        f'params/{pn}_std': p.data.std().log10().item(),
        f'params/{pn}_norm2': p.data.norm(2).mean().log10().item(),
    })
    if p.grad is not None:
      wandb.log({
        f'grads/{pn}_mean': p.grad.mean().item(),
        f'grads/{pn}_std': p.grad.std().log10().item(),
        f'grads/{pn}_norm2': p.grad.norm(2).mean().log10().item(),
        f'grads_over_param/{pn}_mean': (p.grad.mean() / p.data.mean()).item(),
        f'grads_over_param/{pn}_std': (p.grad.std() / p.data.std()).log10().item(),
        f'grads_over_param/{pn}_norm2': (p.grad.norm(2).mean() / p.data.norm(2).mean()).log10().item(),
      })
      if not(train_config.use_schedule_free_lr) and model_args['muP']:
        if p.dim() >=2:
          if p.dim() == 2 and not any(sub in pn for sub in special_weights):
            wandb.log({
              f'lr/{pn}': lr/p.shape[1],
              f'effective_update/{pn}_mean': ((lr/p.shape[1])*p.grad).mean().item(),
              f'effective_update/{pn}_std': ((lr/p.shape[1])*p.grad).std().log10().item(),
              f'effective_update/{pn}_norm2': ((lr/p.shape[1])*p.grad).norm(2).mean().log10().item(),
              f'effective_update_over_param_data/{pn}_mean': ((((lr/p.shape[1])*p.grad).mean())/p.data.mean()).item(),
              f'effective_update_over_param_data/{pn}_std': ((((lr/p.shape[1])*p.grad).std())/p.data.std()).log10().item(),
              f'effective_update_over_param_data/{pn}_norm2': ((((lr/p.shape[1])*p.grad).norm(2).mean())/p.data.norm(2).mean()).log10().item(),
            })
          elif 'shared_soft_experts' in pn:
            wandb.log({
              f'lr/{pn}': lr/p.shape[0],
              f'effective_update/{pn}_mean': ((lr/p.shape[0])*p.grad).mean().item(),
              f'effective_update/{pn}_std': ((lr/p.shape[0])*p.grad).std().log10().item(),
              f'effective_update/{pn}_norm2': ((lr/p.shape[0])*p.grad).norm(2).mean().log10().item(),
              f'effective_update_over_param_data/{pn}_mean': ((((lr/p.shape[0])*p.grad).mean())/p.data.mean()).item(),
              f'effective_update_over_param_data/{pn}_std': ((((lr/p.shape[0])*p.grad).std())/p.data.std()).log10().item(),
              f'effective_update_over_param_data/{pn}_norm2': ((((lr/p.shape[0])*p.grad).norm(2).mean())/p.data.norm(2).mean()).log10().item(),
            })
          else:
            wandb.log({
              f'lr/{pn}': lr,
              f'effective_update/{pn}_mean': (lr*p.grad).mean().item(),
              f'effective_update/{pn}_std': (lr*p.grad).std().log10().item(),
              f'effective_update/{pn}_norm2': (lr*p.grad).norm(2).mean().log10().item(),
              f'effective_update_over_param_data/{pn}_mean': ((lr*p.grad).mean()/p.data.mean()).item(),
              f'effective_update_over_param_data/{pn}_std': ((lr*p.grad).std()/p.data.std()).log10().item(),
              f'effective_update_over_param_data/{pn}_norm2': ((lr*p.grad).norm(2).mean()/p.data.norm(2).mean()).log10().item(),
            })
        else:
          wandb.log({
            f'lr/{pn}': lr,
            f'effective_update/{pn}_mean': (lr*p.grad).mean().item(),
            f'effective_update/{pn}_std': (lr*p.grad).std().log10().item(),
            f'effective_update/{pn}_norm2': (lr*p.grad).norm(2).mean().log10().item(),
            f'effective_update_over_param_data/{pn}_mean': ((lr*p.grad).mean()/p.data.mean()).item(),
            f'effective_update_over_param_data/{pn}_std': ((lr*p.grad).std()/p.data.std()).log10().item(),
            f'effective_update_over_param_data/{pn}_norm2': ((lr*p.grad).norm(2).mean()/p.data.norm(2).mean()).log10().item(),
          })  
      else: # cases : schedule free lr or no muP 
        wandb.log({
          f'lr': lr,
          f'effective_update/{pn}_mean': (lr*p.grad).mean().item(),
          f'effective_update/{pn}_std': (lr*p.grad).std().log10().item(),
          f'effective_update/{pn}_norm2': (lr*p.grad).norm(2).mean().log10().item(),
          f'effective_update_over_param_data/{pn}_mean': ((lr*p.grad).mean()/p.data.mean()).item(),
          f'effective_update_over_param_data/{pn}_std': ((lr*p.grad).std()/p.data.std()).log10().item(),
          f'effective_update_over_param_data/{pn}_norm2': ((lr*p.grad).norm(2).mean()/p.data.norm(2).mean()).log10().item(),
        })      
  
  activations.clear()

@torch.no_grad()
def saveCkpt(best_val_loss: float, 
             model: torch.nn.Module, 
             model_args: Dict, iter: int, 
             config: Dict, filepath: str, 
             optimizer: torch.optim.Optimizer, 
             use_schedule_free_lr: Optional[bool] = False) -> float:
    
    if use_schedule_free_lr:
      optimizer.eval()
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
    print(f"Saved to {file_path}")
    if use_schedule_free_lr:
      optimizer.train()

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model: torch.nn.Module, 
                  val_dataloader: torch.utils.data.DataLoader, 
                  ctx: Union[torch.amp.autocast, nullcontext], 
                  device: str, 
                  optimizer: Optional[torch.optim.Optimizer] = None, 
                  use_schedule_free_lr: Optional[bool] = False) -> float:
    
    model.eval()
    if use_schedule_free_lr:
      assert optimizer != None, "please provide the optimizer when you use estimate_loss with schedule free lr"
      optimizer.eval()
    losses = torch.zeros(len(val_dataloader))
    for step, (X, y) in enumerate(val_dataloader):
      X, Y = X.to(device), y.to(device)
      with ctx: 
        _, loss = model(X, Y) 
        losses[step] = loss.item()
    model.train()
    if use_schedule_free_lr:
      optimizer.train()
    return losses.mean()

def buildModel(init_from: str, 
               device: str, 
               out_dir: str, 
               default_model_args: Dict, 
               ckpt_filename: str,
               meta_vocab_size: Optional[int] = None):
  
  if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    default_model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    model_args = default_model_args
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
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
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
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


# decay scheduler (cosine with warmup)
def cos_schedule(current_step: int, warmup_iters: int, decay_iters: int, min_value: float, max_value: float) -> float:
  # 1) linear warmup for warmup_iters steps
  if current_step < warmup_iters:
    return max_value * current_step / warmup_iters
  # 2) if current_step > decay_iters, return min_value
  if current_step > decay_iters:
    return min_value
  # 3) in between, use cosine decay down to min_value # count steps from the initiation of the cos decay
  decay_ratio = (current_step - warmup_iters) / (decay_iters - warmup_iters)
  assert 0.0 <= decay_ratio <= 1.0, "something wrong with the cosine schedule config"
  coeff = (1/2) * (1.0 + math.cos(math.pi * decay_ratio)) 
  return max_value + coeff * (min_value - max_value)


# decay scheduler (linear with warmup)
def linear_schedule(current_step: int, warmup_iters: int, decay_iters: int, min_value: float, max_value: float):
  # 1) linear warmup for warmup_iters steps
  if current_step < warmup_iters:
    return max_value * current_step / warmup_iters
  # 2) if current_step > decay_iters, return min_value
  if current_step > decay_iters:
    return min_value
  # 3) in between, use linear decay down to min_value
  coeff = (current_step - warmup_iters) / (decay_iters - warmup_iters) # starts with 0 and ends with 1
  return max_value + coeff * (min_value - max_value)

def calc_lr(train_config: TRAIN_CONFIG, current_step: int, max_iters: int):
  linear_schedule_kwargs = {
    "current_step": current_step, 
    "warmup_iters": train_config.warmup_iters * max_iters, 
    "decay_iters": train_config.lr_decay_iters * max_iters, 
    "min_value": train_config.min_lr, 
    "max_value": train_config.learning_rate
  }
  if train_config.lr_schedule == 'constant':
    return train_config.learning_rate
  elif train_config.lr_schedule == 'cos_schedule':
    return cos_schedule(**linear_schedule_kwargs)
  elif train_config.lr_schedule == 'linear_schedule':
    return linear_schedule(**linear_schedule_kwargs)
  else:
    raise ValueError("You must set a lr schedule between constant/cos_schedule/linear_schedule")


# Note: We're using the pin_memory=True parameter in the create_dataloaders() function to speed up computation. pin_memory=True avoids unnecessary
# copying of memory between the CPU and GPU memory by "pinning" examples that have been seen before. Though the benefits of this will likely be seen
# with larger dataset sizes

def buildDataBlock(train_data: np.ndarray, block_size: int, limitSizeTo: int = None, stepSize: int = 1):
  maxElm = (min(limitSizeTo, len(train_data) - block_size)) if limitSizeTo else  (len(train_data) - block_size)
  X = [torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in range(0, maxElm, stepSize)] 
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

