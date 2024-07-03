from dataclasses import dataclass

from efficient_transformer_v1.model import AnyModalMirasolConfig

@dataclass
class TRAIN_CONFIG:
  seed: int = 42
  dataset: str = "data/shakespear_char/" # "shakespear_char"
  model_config: AnyModalMirasolConfig = AnyModalMirasolConfig(block_size=768, vocab_size=None, n_layer_enc_dec=2, n_head=8, n_embd = [64, 128, 256],
                                      dropout=0.1, enc_dropout = 0.1, enc_type = 'tranfo_enc', bias=False, activation = 'gelu', normalization = 'rmsnorm',
                                      noisy_embd_alpha = 0.0, nb_mem_token = 0, muP=True, apply_gradient_checkpointing = False, gradient_checkpoint_sequential_factor = 2,
                                      pos_embd ='simple', n_groups = 16, latent_size = [16, 2], n_block_local_global_causal_sa = 2,# 'simple','no'
                                      n_block_latent_attn = 16, n_block_transformer_enc = 2, interleave_local_global_csa = False,
                                      lernable_skip_hiearchy = False, latent_causal_loss_type = None,
                                      nb_soft_experts = 12, nb_shared_experts = 1, use_same_kv_for_soft_experts = False, apply_soft_experts_every = 4,
                                      init_weight = 'spike_no_more', coeff_soft_expert_mlp_dim = 1, apply_soft_experts_in_blocks = ['latent'], use_dense_former_in_latent=False,
                                      use_pos_embd_in_enc_dec = True, up_sampling_type = 'repeat') # config used when init from scratch
  compile: bool = True # use PyTorch 2.0 to compile the model to be faster
  out_dir: str = "out"
  wandb_log: bool = False # noticeable additional latency. comment some of the info save in train_utils.py within writeLogs to save time
  wandb_project: str = 'zero'
  wandb_run_name: str = 'anymodal_mirasol'
  eval_interval: int = 1
  log_interval: int = 1
  restrict_val_iters: int = None # 20000 # set to None if want to eval on the whole val set
  restrict_eval_iters: int = None # 20000 # set to None if want to eval on the whole eval set
  step_size_eval: int = 32
  always_save_checkpoint: bool = False # if True, always save a checkpoint after each eval
  init_from: str = 'scratch' # 'scratch' or 'resume'
  num_epochs: int = 1
  optimizer: str = 'adamw'
  use_8_bit_optim: bool = False # doesn't support schedule free lr # AdamW8bit https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim
  lr_schedule: str = 'constant' # constant, cos_schedule, linear_schedule
  use_schedule_free_lr: bool = False # https://github.com/facebookresearch/schedule_free
  learning_rate: float = 3.3e-1 # try 3.5e-3 with schedule free # max learning rate 
  weight_decay: float = 0.01
  beta1: float = 0.9 # 0.98 schedule free lr (recomended to tune)
  beta2: float = 0.95 # 0.999 schedule free lr
  grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0
  warmup_iters: float = 0.1 # how many steps to warm up for # percentage of total step
  lr_decay_iters: float = 0.95 # should be ~= max_steps per Chinchilla # percentage of total step
  min_lr: float = 4.8e-2 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
  batch_size: int = 64 # min train batch size # if gradient_accumulation_steps > 1, this is the micro-batch size
  gradient_accumulation_steps: int = 2 * 48 # used to simulate larger batch sizes
  nb_grad_update_to_register_stats: int = 50
  linearBSIcr: bool = False # not implemented yet
  startStepBsSchedule: int = 10
  maxBS: int = 160
  coeffBSIcr: int = 2
  icrIntervalBsSchedule: int = 200
  val_batch_size: int = 2 * 64
  block_size: int = 768 # context_length
  latent_causal_modeling_loss_coeff: float = 0.003 # should be between 0.0 and 1
  latent_causal_loss_warmup_max_step: float = 0.20 # x % dataset to warm until max coeff latent loss  