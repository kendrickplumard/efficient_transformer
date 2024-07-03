from dataclasses import dataclass, field

from efficient_transformer_v2.model import ConditionalTFConfig


@dataclass
class TRAIN_CONFIG:
  seed: int = 42
  dataset: str = "shakespear_char" # "shakespear_char"
  model_config: ConditionalTFConfig = ConditionalTFConfig(block_size=768, vocab_size=None, n_series_blocks=3, n_light_heavy_blocks_in_each_serie=2, n_head=32, n_embd=256,
                                                          light_branch_query_capacity=0.12, heavy_branch_query_capacity=0.06, use_separate_routers_for_q_and_kv=False, 
                                                          light_branch_key_value_capacity=0.12, heavy_branch_key_value_capacity=0.06, coeff_light_branch_n_embd=1./2.,
                                                          coeff_heavy_branch_n_embd=2.0, use_same_kv_for_soft_experts=False, use_light_heavy_branch_at_same_layer_level=True,
                                                          nb_soft_experts=6, nb_shared_experts=2, coeff_soft_expert_mlp_dim=1, n_groups=16, dropout=0.0, bias=False, 
                                                          activation='gelu', relufication=False, normalization='rmsnorm', noisy_embd_alpha=0.0, nb_mem_token=0, muP=True, 
                                                          pos_embd='simple', init_weight='nanogpt', apply_gradient_checkpointing=False, gradient_checkpoint_sequential_factor=2) # config used when init from scratch
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
  use_8_bit_optim: list[bool] = field(default_factory=lambda: [False, True]) # doesn't support schedule free lr # AdamW8bit https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim
  lr_schedule: list[str] = field(default_factory=lambda: ['constant', 'constant']) # constant, cos_schedule, linear_schedule 
  use_schedule_free_lr: bool = False # https://github.com/facebookresearch/schedule_free
  learning_rate: list[float] = field(default_factory=lambda: [1.2e-1, 1.2e-1]) # max learning rate # regular model param and samping predictor param
  max_steps: int = 5000 # total number of training iterations
  weight_decay: list[float] = field(default_factory=lambda: [0.01, 0.01]) # regular model param and samping predictor param
  beta1: list[float] = field(default_factory=lambda: [0.9, 0.9]) # regular model param and samping predictor param
  beta2: list[float] = field(default_factory=lambda: [0.95, 0.95]) # regular model param and samping predictor param
  grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0
  warmup_iters: list[float] = field(default_factory=lambda: [0.1, 0.1]) # how many steps to warm up for # percentage of total step
  lr_decay_iters: list[float] = field(default_factory=lambda: [0.95, 0.95]) # should be ~= max_steps per Chinchilla # percentage of total step
  min_lr: list[float] = field(default_factory=lambda: [1.2e-2, 1.2e-2]) # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
  batch_size: int = 64 # 96 # min train batch size # 128 # if gradient_accumulation_steps > 1, this is the micro-batch size
  gradient_accumulation_steps: int = 2 * 48 # used to simulate larger batch sizes
  linearBSIcr: bool = True
  startStep: int = 10
  maxBS: int = 160
  coeffBSIcr: int = 2
  icrInterval: int = 200
  val_batch_size: int = 2 * 64
  block_size: int = 768 # context_length
