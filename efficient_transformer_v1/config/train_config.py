from dataclasses import dataclass

from efficient_transformer_v1.model import AnyModalMirasolConfig

@dataclass
class TRAIN_CONFIG:
  seed: int = 42
  dataset: str = "data/shakespear_char/" # "shakespear_char"
  model_config: AnyModalMirasolConfig = AnyModalMirasolConfig(block_size=768, vocab_size=None, n_layer_enc_dec=2, n_head=8, n_embd = [64, 128, 256],
                                      dropout=0.1, enc_dropout = 0.1, enc_type = 'tranfo_enc', bias=False, activation = 'gelu', normalization = 'rmsnorm',
                                      noisy_embd_alpha = 0.0, nb_mem_token = 0, muP=False, apply_gradient_checkpointing = False, gradient_checkpoint_sequential_factor = 2,
                                      pos_embd ='simple', n_groups = 16, latent_size = [16, 2], n_block_local_global_causal_sa = 2,# 'simple','no'
                                      n_block_latent_attn = 16, n_block_transformer_enc = 2, interleave_local_global_csa = False,
                                      lernable_skip_hiearchy = False, latent_causal_loss_type = None,
                                      nb_soft_experts = 12, nb_shared_experts = 1, use_same_key_for_soft_experts = False, apply_soft_experts_every = 4,
                                      init_weight = 'nanogpt', coeff_soft_expert_mlp_dim = 1, apply_soft_experts_in_blocks = ['latent'], use_dense_former_in_latent=False,
                                      use_pos_embd_in_enc_dec = True, up_sampling_type = 'repeat') # config used when init from scratch
  compile: bool = True # use PyTorch 2.0 to compile the model to be faster
  out_dir: str = "out"
  log_dir: str = "log_dir"
  eval_interval: int = 1
  log_interval: int = 1
  restrict_val_iters: int = None # 20000 # set to None if want to eval on the whole val set
  restrict_eval_iters: int = None # 20000 # set to None if want to eval on the whole eval set
  step_size_eval: int = 32
  always_save_checkpoint: bool = True # if True, always save a checkpoint after each eval
  init_from: str = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
  num_epochs: int = 2
  optimizer: str = 'adamw'
  lr_schedule: str = 'linear_schedule' # constant, cos_schedule, linear_schedule
  learning_rate: float = 4.8e-4 # 6e-4 # 6e-2 # max learning rate # muP decreases the lr too much
  # max_steps: int = 5000 # total number of training iterations
  weight_decay: float = 0.25
  beta1: float = 0.9
  beta2: float = 0.95
  grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0
  decay_lr: bool = False # whether to decay the learning rate
  warmup_iters: float = 0.5 # how many steps to warm up for # percentage of total step
  lr_decay_iters: float = 0.85 # should be ~= max_steps per Chinchilla # percentage of total step
  min_lr: float = 4.8e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
  batch_size: int = 64 #96 # min train batch size # 128 # if gradient_accumulation_steps > 1, this is the micro-batch size
  gradient_accumulation_steps: int = 2 * 68 # 2 * 196 # 2 * 224 # 5 * 224 # used to simulate larger batch sizes
  nb_grad_update_to_register_stats: int = 50
  register_stats_wb: bool = False
  linearBSIcr: bool = True
  startStep: int = 10
  maxBS: int = 160
  coeffBSIcr: int = 2
  icrInterval: int = 200
  val_batch_size: int = 2 * 64
  block_size: int = 768 # context_length
  latent_causal_modeling_loss_coeff: float = 0.003 # 0.2 # should be between 0.0 and 1
  latent_causal_loss_warmup_max_step: float = 0.20 # 20 % dataset to warm until max coeff latent loss