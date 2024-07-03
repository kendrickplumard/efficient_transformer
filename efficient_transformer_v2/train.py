import argparse
import atexit
import copy
import os
import pickle
import schedulefree
import time
import torch
import torch.nn as nn
import wandb

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from efficient_transformer_v2.model import GatedActivation
from efficient_transformer_v2.train_utils import *

torch._dynamo.config.suppress_errors = True
torch.autograd.set_detect_anomaly(True)

main_timer = time.time()

train_config = TRAIN_CONFIG()
tokens_per_iter = train_config.gradient_accumulation_steps * train_config.maxBS * train_config.model_config.block_size
print(f"tokens per iteration will be at most: {tokens_per_iter:,}")


os.makedirs(train_config.out_dir, exist_ok=True)
ckpt_filename = f'conditional_tf.pt' # ici utilise default config si from resume et entre temps default config a changé nom pas approprié
config = copy.deepcopy(vars(train_config))
config['model_config'] = copy.deepcopy(vars(config['model_config'])) # get a dict from the GPTConfig instance
torch.manual_seed(train_config.seed)
torch.cuda.manual_seed(train_config.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

train_dataset = CustomDataset(os.path.join(train_config.dataset, 'train.bin'), train_config.model_config.block_size)
val_dataset = CustomDataset(os.path.join(train_config.dataset, 'val.bin'),
                            train_config.model_config.block_size,
                            train_config.restrict_val_iters)
train_dataloader = DataLoader(dataset = train_dataset,
                              batch_size = train_config.batch_size,
                              num_workers = os.cpu_count(),
                              prefetch_factor=4,
                              shuffle = True,
                              pin_memory = True,
                              drop_last = True)
val_dataloader = DataLoader(dataset = val_dataset,
                            batch_size = train_config.val_batch_size,
                            num_workers = os.cpu_count(),
                            prefetch_factor=4,
                            shuffle = False,
                            pin_memory = True)
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
model, model_args, checkpoint, iter, val_loss_last_ckpt = buildModel(train_config.init_from, device, meta_vocab_size, train_config.out_dir, copy.deepcopy(vars(train_config.model_config)), ckpt_filename)
if train_config.model_config.block_size < model.config.block_size:
    model.crop_block_size(train_config.model_config.block_size)
    model_args['block_size'] = train_config.model_config.block_size # so that the checkpoint will have the right value
model.to(device)

# init model stats
if train_config.wandb_log:
  activations = {}
  # define activation hook function
  def hook_fn(module, input, output):
    activations[module]= output.detach()

  # attach hook
  for name, module in model.named_modules():
    if isinstance(module, nn.GELU) or isinstance(module,  GatedActivation):
      module.register_forward_hook(hook_fn)

  wandb.init(project=train_config.wandb_project, name=train_config.wandb_run_name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S"), config=vars(train_config))

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# training loop
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
lr = train_config.learning_rate[0]
lr_sampling_predictor = train_config.learning_rate[1]
train_perplexity = []
start_epoch = iter // len(train_dataloader)
max_iters = len(train_dataloader) * train_config.num_epochs

# optimizer
print(f"using 8 bit optimizer for regular params: {train_config.use_8_bit_optim[0]}")
print(f"using 8 bit optimizer for sampling predictor params: {train_config.use_8_bit_optim[1]}")
if True in train_config.use_8_bit_optim:
  assert (train_config.compile), "8 bit Adam requires the compile flag"

print(f"using schedule free lr: {train_config.use_schedule_free_lr}")
if train_config.use_schedule_free_lr:
  assert not(model_args['muP']), "cannot combine schedule free lr with muP. unspecified behavior"
  assert not(True in train_config.use_8_bit_optim), "schedule free lr doesn't support the ADAMW8Bit optimizer currently"
  optimizer = schedulefree.AdamWScheduleFree([p for pn, p in model.named_parameters() if 'sampling_predictor' not in pn],
                                             lr=train_config.learning_rate[0],
                                             betas=(train_config.beta1[0], train_config.beta2[0]),
                                             weight_decay=train_config.weight_decay[0],
                                             warmup_steps=int((train_config.warmup_iters[0] * max_iters) // train_config.gradient_accumulation_steps)) # they update warmup counter every optimizer step

  optimizer_sampling_predictor = schedulefree.AdamWScheduleFree([p for pn, p in model.named_parameters() if 'sampling_predictor' in pn],
                                                                lr=train_config.learning_rate[1],
                                                                betas=(train_config.beta1[1], train_config.beta2[1]),
                                                                weight_decay=train_config.weight_decay[1],
                                                                warmup_steps=int((train_config.warmup_iters[1] * max_iters) // train_config.gradient_accumulation_steps)) 
else:
  optimizer, optimizer_sampling_predictor = model.configure_optimizers(train_config.weight_decay,
                                                                       train_config.learning_rate,
                                                                      [(train_config.beta1[0], train_config.beta2[0]), (train_config.beta1[1], train_config.beta2[1])], 
                                                                       train_config.use_8_bit_optim,
                                                                       device)

if train_config.init_from == 'resume':
  optimizer.load_state_dict(checkpoint['optimizer'])
  optimizer_sampling_predictor.load_state_dict(checkpoint['optimizer_sampling_predictor'])
checkpoint = None # free up memory

# compile the model
if train_config.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

# cleaning
def cleaning():
  train_dataloader = None
  val_dataloader = None
  if train_config.wandb_log:
    wandb.finish()
atexit.register(cleaning)


for epoch in range(start_epoch, train_config.num_epochs):

  # forward backward update, with optional gradient accumulation to simulate larger batch size
  # and using the GradScaler if data type is float16
  model.train()
  if train_config.use_schedule_free_lr:
    optimizer.train()
    optimizer_sampling_predictor.train()
  total_loss = 0.0
  total_length = len(train_dataloader) // train_config.gradient_accumulation_steps + int (len(train_dataloader) % train_config.gradient_accumulation_steps != 0)
  pbar = tqdm(colour="blue", desc=f"Training epoch: {epoch + 1}", total = total_length, dynamic_ncols = True)
  for step, (X, y) in enumerate(train_dataloader):

    X, y = X.to(device), y.to(device)
    with ctx:
      logits, losses = model(X, y)
      losses = {key: losses[key] / train_config.gradient_accumulation_steps for key in losses} # scale the loss to account for gradient accumulation
    total_loss += losses['CE'].detach().float()
    loss=sum([losses[key] for key in losses if losses[key].requires_grad]) # filter eventual key_value_sampling_predictor if not used
    scaler.scale(loss).backward(retain_graph=False)
    if (step + 1) % train_config.gradient_accumulation_steps == 0 or step == len(train_dataloader) -1:
      # clip the gradient
      if train_config.grad_clip != 0.0:
        scaler.unscale_(optimizer) 
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

      # update lr when we are about to step the optimizer
      if not(train_config.use_schedule_free_lr):
        # determine and set the learning rate for this iteration
        lr = calc_lr(train_config, (epoch + 1) * (step + 1), max_iters)
        for param_group in optimizer.param_groups: #
          param_group['lr'] = lr / param_group['muPScale'] if model_args['muP'] else lr

        lr_sampling_predictor = calc_lr(train_config, (epoch + 1) * (step + 1), max_iters, position=1)
        for param_group in optimizer_sampling_predictor.param_groups: #
          param_group['lr'] = lr_sampling_predictor / param_group['muPScale'] if model_args['muP'] else lr_sampling_predictor

      # step the optimizer and scaler if training in fp16
      scaler.step(optimizer) 
      scaler.step(optimizer_sampling_predictor)
      scaler.update()

      if (step + 1) % train_config.eval_interval == 0:
        val_loss = estimate_loss(model, val_dataloader, ctx, device, [optimizer, optimizer_sampling_predictor], train_config.use_schedule_free_lr)
        train_loss = {key: losses[key].detach().float() * train_config.gradient_accumulation_steps for key in losses}
        sampling_predictors_mean_train_loss = [train_loss[key] for key in train_loss if key != 'CE']
        sampling_predictors_mean_train_loss = sum(sampling_predictors_mean_train_loss) / len(sampling_predictors_mean_train_loss)
        print(f"\nstep {step + 1}, "
              f"train loss {train_loss['CE']:.4f}, "
              f"val loss  {val_loss['CE']:.4f}, \n"
              f"sampling predictors mean train loss {sampling_predictors_mean_train_loss:.4f}, "
              f"sampling predictors mean val loss {val_loss['sampling_predictors_mean_loss']:.4f}")

        if val_loss['CE'] < val_loss_last_ckpt['CE'] or train_config.always_save_checkpoint:
          val_loss_last_ckpt = val_loss
          saveCkpt(val_loss, 
                   model, 
                   model_args, 
                    (step + 1) * (epoch + 1), 
                   config, 
                   os.path.join(train_config.out_dir, ckpt_filename), 
                   [optimizer, optimizer_sampling_predictor], 
                   train_config.use_schedule_free_lr)

        if train_config.wandb_log: # grad already unscaled
          writeLogs(model,
                    activations,
                    {'train_loss': train_loss, 'val_loss': val_loss},
                    (step + 1) * (epoch + 1),
                    [lr, lr_sampling_predictor],
                    [optimizer, optimizer_sampling_predictor],
                    train_config,
                    model_args)

      # flush the gradients as soon as we can, no need for this memory anymore
      optimizer.zero_grad(set_to_none=True)
      optimizer_sampling_predictor.zero_grad(set_to_none=True)
      pbar.update(1)

      
    # t1 = time.time()
    # dt = t1 - t0
    # t0 = t1
    # if local_iter_num >= 5: # let the training loop settle a bit
    #   mfu = model.estimate_mfu(train_config.batch_size, dt) # juste bs plus de * grad_acc
    #   running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
    #   print(f"mfu: {running_mfu:.4f}")
    # local_iter_num += 1
  train_perplexity.append(float(torch.exp(total_loss / len(train_dataloader))))
  pbar.close()


val_dataloader = None
train_dataloader = None
if train_config.wandb_log:
    wandb.finish()

print(f'\n avg train perplexity: {sum(train_perplexity)/len(train_perplexity):.4f}')
print('val loss of the last checkpoint registered: ', val_loss_last_ckpt['CE'].item())
print(f"\ntime elapsed: {(time.time() - main_timer) / 60:.2f} min")
