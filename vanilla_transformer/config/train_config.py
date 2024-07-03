from dataclasses import dataclass

from vanilla_transformer.model import GPTConfig

@dataclass
class TRAIN_CONFIG:
  seed: int = 42
  dataset: str = "data/shakespear_char/" # "shakespear_char"
  model_config: GPTConfig = GPTConfig(block_size=768, vocab_size=None, n_layer=26, n_head=8, n_embd=256,
                                      dropout=0.1, bias=False, activation = 'gelu', normalization = 'rmsnorm',
                                      noisy_embd_alpha = 0.0, nb_mem_token = 0, muP=True,
                                      apply_gradient_checkpointing = False) # config used when init from scratch
  compile: bool = True # use PyTorch 2.0 to compile the model to be faster
  out_dir: str = "out"
  wandb_log: bool = False # noticeable additional latency. comment some of the info save in train_utils.py within writeLogs to save time
  wandb_project: str = 'zero'
  wandb_run_name: str = 'gpt'
  eval_interval: int = 1
  log_interval: int = 1
  restrict_val_iters: int = None # 20000 # set to None if want to eval on the whole val set
  restrict_eval_iters: int = None # 20000 # set to None if want to eval on the whole eval set
  step_size_eval: int = 32
  always_save_checkpoint: bool = False # if True, always save a checkpoint after each eval
  init_from: str = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
  num_epochs: int = 1
  optimizer: str = 'adamw'
  use_8_bit_optim: bool = False # doesn't support schedule free lr # AdamW8bit https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim
  lr_schedule: str = 'constant' # constant, cos_schedule, linear_schedule
  use_schedule_free_lr: bool = False # https://github.com/facebookresearch/schedule_free
  learning_rate: float = 2.1e-1 # max learning rate # muP decreases the lr too much
  max_steps: int = 5000 # total number of training iterations
  weight_decay: float = 0.01
  beta1: float = 0.9
  beta2: float = 0.95
  grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0
  warmup_iters: float = 0.1 # how many steps to warm up for # percentage of total step
  lr_decay_iters: float = 0.95 # should be ~= max_steps per Chinchilla # percentage of total step
  min_lr: float = 2.1e-2 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
  batch_size: int = 64 # 96 # min train batch size # 128 # if gradient_accumulation_steps > 1, this is the micro-batch size
  gradient_accumulation_steps: int = 2 * 48 # used to simulate larger batch sizes
  linearBSIcr: bool = True # not implemente yet
  startStep: int = 10
  maxBS: int = 160
  coeffBSIcr: int = 2
  icrInterval: int = 200
  val_batch_size: int = 2 * 64
  block_size: int = 768 # context_length
