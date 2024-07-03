import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

from dataclasses import dataclass, field
from torchao.prototype.low_bit_optim import AdamW8bit
from torch.utils.checkpoint import checkpoint_sequential

class SwiGLU(nn.Module):
  """ Noam Shazeer SwiGLU activation, better in Transformers MLP than others """
  def forward(self, x):
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x

class RMSNorm(torch.nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim)) # gain param Learnable scaling parameter.

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    return self._norm(x.float()).type_as(x) * self.weight

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class LatentCausalReconstructionLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, predictions, targets):
    return 1.0 - F.cosine_similarity(predictions, targets, dim = -1).mean()

class LatentCausalSigLipLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.t_prime = nn.Parameter(torch.tensor(10.0, dtype=torch.float), requires_grad = True)
    self.b = nn.Parameter(torch.tensor(-10.0, dtype=torch.float), requires_grad = True)

  def forward(self, predictions, targets):
    B = predictions.size()[0]
    T = self.t_prime.exp()
    predictions_normalized = F.normalize(predictions, p = 2, dim = -1) # B, T-1, C
    targets_normalized = F.normalize(targets, p = 2, dim = -1) # B, T-1, C
    logits = (predictions_normalized @ targets_normalized.transpose(1, 2)) * T + self.b # B, T-1, T-1
    labels = 2 * torch.eye(B) - torch.ones(B, B) # B, B
    return -F.logsigmoid(labels * logits).sum() / B

def compute_attn(q, k, v, attn_dropout, use_flash_attn, config, training, bias_mask):
  if use_flash_attn:
    # efficient attention using Flash Attention CUDA kernels
    scale = (1.0 / k.size(-1)) if config.muP else (1.0 / math.sqrt(k.size(-1)))
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=config.dropout \
                                                                 if training else 0, is_causal=True, scale=scale)
  else:
    # manual implementation of attention
    att = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1)) if config.muP else (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(bias_mask, -torch.finfo(q.dtype).max)
    att = F.softmax(att, dim=-1)
    att = attn_dropout(att)
    y = att @ v
  return y

class PerceiverEncoder(nn.Module):

    def __init__(self, config, idx_n_embd):
        super().__init__()
        assert config.n_embd[idx_n_embd] % config.n_head == 0
        self.config = config
        self.in_n_embd = config.n_embd[idx_n_embd]
        self.out_n_embd = config.n_embd[idx_n_embd + 1]
        # query, key, value projections for all heads, but in a batch
        self.enc_to_q = nn.Linear(self.in_n_embd, self.out_n_embd, bias=config.bias)
        self.enc_to_kv = nn.Linear(self.in_n_embd, 2 * self.out_n_embd, bias=config.bias)
        # output projection
        self.enc_proj_to_embd = nn.Linear(self.out_n_embd, self.out_n_embd, bias=config.bias) # modify here if i want diff dim from wte and first enc, can set the out dim here or directly at to_q. to be tested
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.latent_size = config.latent_size[idx_n_embd]
        self.n_groups = self.config.n_groups
        self.ln_qk = RMSNorm(self.out_n_embd) if config.normalization == 'rmsnorm' else LayerNorm(self.out_n_embd, bias=config.bias)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')


    def forward(self, x, local_csa_mem_tokens = None, mem_tokens = None):
        B, ng, lg, C = x.size() # batch size, n_groups, length group, embedding dimensionality (n_embd)
        assert lg >= self.latent_size, "should have at least ls token in the input in order to reduce the size"

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        assert self.out_n_embd % self.n_head == 0, f"n_embd ({self.out_n_embd}) must be a multiple of n_head ({self.n_head})"
        k, v  = self.enc_to_kv(x).split(self.out_n_embd, dim=3) # B, ng, lg, out_n_embd
        q = self.enc_to_q(x[:, :, -self.latent_size:, :]) # B, ng, ls, out_n_embd
        q, k = self.ln_qk(q), self.ln_qk(k)
        q = q.view(B, ng, self.latent_size, self.n_head, self.out_n_embd // self.n_head).transpose(2, 3) # (B, ng, nh, ls, hs)
        k = k.view(B, ng, lg, self.n_head, self.out_n_embd // self.n_head).transpose(2, 3) # (B, ng, nh, lg, hs)
        v = v.view(B, ng, lg, self.n_head, self.out_n_embd // self.n_head).transpose(2, 3) # (B, ng, nh, lg, hs)
        attn_mask = torch.triu(torch.ones((self.latent_size, lg), device = x.device, dtype = torch.bool), # causal mask to ensure that attention is only applied to the last ls tokens
                                    diagonal = lg - self.latent_size + 1) \
                                    .view(1, 1, 1, self.latent_size, lg) # mask one above diag with regular attn, now mask M - N + 1 above diag


        # don't need to send a copy of q, k , v # T/lg must be smaller or equal to block_size # (B, ng, nh, ls, hs)
        y = compute_attn(q, k, v, self.attn_dropout, self.flash, self.config, self.training, attn_mask[:, :, :, :self.latent_size, :lg])
        y = y.transpose(2, 3).contiguous().view(B, ng, self.latent_size, self.out_n_embd) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.enc_proj_to_embd(y))
        if mem_tokens != None:
          y = [self.enc_to_q(mem_tokens), self.enc_to_q(local_csa_mem_tokens), y] # recycle to_q to proj mem_tokens

        # output projection
        return mem_tokens,local_csa_mem_tokens, y # (B, ng, ls, out_n_embd), (B, nb_mem_tokens, out_n_embd)

class LocalCausalSelfAttention(nn.Module):

    def __init__(self, config, in_n_embd, out_n_embd, use_soft_experts = False):
        super().__init__()
        assert out_n_embd % config.n_head == 0
        self.in_n_embd = in_n_embd
        self.out_n_embd = out_n_embd
        self.config = config
        self.use_soft_experts = use_soft_experts
        self.nb_non_shared_soft_experts = config.nb_soft_experts - config.nb_shared_experts
        # key, query, value projections for all heads, but in a batch
        if not use_soft_experts:
          self.csa_to_qkv = nn.Linear(self.in_n_embd, 3 * self.out_n_embd, bias=config.bias)
        else:
          if self.config.use_same_kv_for_soft_experts:
            self.csa_to_q = nn.Linear(self.in_n_embd, self.nb_non_shared_soft_experts * self.out_n_embd, bias=config.bias)
            self.csa_to_kv = nn.Linear(self.in_n_embd, 2 * self.out_n_embd, bias=config.bias)
          else:
            self.csa_to_qkv = nn.Linear(self.in_n_embd, 3 * (self.out_n_embd * self.nb_non_shared_soft_experts), bias=config.bias)

        # output projection
        self.csa_proj_to_embd = nn.Linear(self.out_n_embd, self.out_n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.ln_qk = RMSNorm(self.out_n_embd) if config.normalization == 'rmsnorm' else LayerNorm(self.out_n_embd, bias=config.bias)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            if not use_soft_experts:
              self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, 1, config.block_size, config.block_size))
            else:
              self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, 1, 1, config.block_size, config.block_size))


    def compute_qkv(self, x):
      B, ng, ls, C = x.size()
      if not self.use_soft_experts:
        q, k, v  = self.csa_to_qkv(x).split(self.out_n_embd, dim=-1) # B, ng, ls, n_embd # local mem tok will also be up dim here if any
        # q, k layer norm before attention
        q, k = self.ln_qk(q), self.ln_qk(k)
        q = q.view(B, ng, ls, self.n_head, self.out_n_embd // self.n_head).transpose(2, 3) # (B, ng, nh, ls, hs)
        k = k.view(B, ng, ls, self.n_head, self.out_n_embd // self.n_head).transpose(2, 3) # (B, ng, nh, ls, hs)
        v = v.view(B, ng, ls, self.n_head, self.out_n_embd // self.n_head).transpose(2, 3) # (B, ng, nh, ls, hs)
      else:
        if self.config.use_same_kv_for_soft_experts:
          k, v  = self.csa_to_kv(x).split(self.out_n_embd, dim=-1) # B, ng, ls, n_embd # local mem tok will also be up dim here if any
          q = self.csa_to_q(x).view(B, ng, ls, self.nb_non_shared_soft_experts, self.out_n_embd).transpose(2, 3) # B, ng, nse, ls, n_embd
          # q, k layer norm before attention
          q, k = self.ln_qk(q), self.ln_qk(k)
          q = q.view(B, ng, self.nb_non_shared_soft_experts, ls, self.n_head, self.out_n_embd // self.n_head).transpose(3, 4) # B, ng, nse, nh, ls, hs
          k = k.view(B, ng, ls, self.n_head, self.out_n_embd // self.n_head).transpose(2, 3).unsqueeze(2) # B, ng, 1, nh, ls, hs)
          v = v.view(B, ng, ls, self.n_head, self.out_n_embd // self.n_head).transpose(2, 3).unsqueeze(2) # (B, ng, 1, nh, ls, hs)
        else:
          q, k, v = self.csa_to_qkv(x).split(self.nb_non_shared_soft_experts * self.out_n_embd, dim=-1) # B, T, nse * n_embd
          q = q.view(B, ng, ls, self.nb_non_shared_soft_experts, self.out_n_embd).transpose(2, 3) # B, ng, nse, ls, n_embd
          k = k.view(B, ng, ls, self.nb_non_shared_soft_experts, self.out_n_embd).transpose(2, 3) # B, ng, nse, ls, n_embd
          v = v.view(B, ng, ls, self.nb_non_shared_soft_experts, self.out_n_embd).transpose(2, 3) # B, ng, nse, ls, n_embd
          # q, k layer norm before attention
          q, k = self.ln_qk(q), self.ln_qk(k)
          q = q.view(B, ng, self.nb_non_shared_soft_experts, ls, self.n_head, self.out_n_embd // self.n_head).transpose(3, 4) # B, ng, nse, nh, ls, hs
          k = k.view(B, ng, self.nb_non_shared_soft_experts, ls, self.n_head, self.out_n_embd // self.n_head).transpose(3, 4) # B, ng, nse, nh, ls, hs
          v = v.view(B, ng, self.nb_non_shared_soft_experts, ls, self.n_head, self.out_n_embd // self.n_head).transpose(3, 4) # B, ng, nse, nh, ls, hs
      return q, k, v



    def forward(self, x, local_csa_mem_tokens = None):
        B, ng, ls, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # first handle mem token if any
        if local_csa_mem_tokens != None:
          local_csa_mem_tokens = local_csa_mem_tokens.view(B, ng, self.config.nb_mem_token, C)
          x = torch.cat((local_csa_mem_tokens, x), dim=2) # B, ng, ls + nb_mem_token, C
          ls += self.config.nb_mem_token


        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        assert self.out_n_embd % self.n_head == 0, f"n_embd ({self.out_n_embd}) must be a multiple of n_head ({self.n_head})"
        # don't need to send a copy bc args aren't modified
        q, k, v = self.compute_qkv(x)

        # (B, ng, nh, ls, ls) if not self.soft_experts (B, ng, nse, nh, ls, ls) if self.soft_experts
        if not self.flash:
          # B, ng, nh, ls, ls or  B, ng, nse, nh, ls, ls
          bias_mask = self.bias[:, :, :, :ls, :ls] == 0 if not self.use_soft_experts else self.bias[:, :, :, :, :ls, :ls] == 0
        else:
          bias_mask = None
        # don't need to send a copy of q, k , v
        y = compute_attn(q, k, v, self.attn_dropout, self.flash, self.config, self.training, bias_mask)
        if not self.use_soft_experts:
          # (B, ng, ls, C)
          y = y.transpose(2, 3).contiguous().view(B, ng, ls, self.out_n_embd) # re-assemble all head outputs side by side
        else:
          # (B, ng, nse, ls, C)
          y = y.transpose(3, 4).contiguous().view(B, ng, self.nb_non_shared_soft_experts, ls, self.out_n_embd) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.csa_proj_to_embd(y))
        if local_csa_mem_tokens != None:
          if not self.use_soft_experts:
            local_csa_mem_tokens, y =  y[:, :, :self.config.nb_mem_token, :].contiguous().view(B, ng * self.config.nb_mem_token, \
                                                                                               self.out_n_embd), y[:, :, self.config.nb_mem_token:, :].contiguous()
          else:
            local_csa_mem_tokens, y =  y[:, :, :, :self.config.nb_mem_token, :].contiguous().view(B, ng * self.config.nb_mem_token, \
                                                                                                  self.nb_non_shared_soft_experts, self.out_n_embd), \
                                                                                                  y[:, :, :, self.config.nb_mem_token:, :].contiguous()

        # output projection
        return local_csa_mem_tokens, y # B, ng, ls, n_embd or B, ng, nse, ls, n_embd


class GlobalCausalSelfAttention(nn.Module):

    def __init__(self, config, n_embd, use_soft_experts = False):
        super().__init__()
        assert n_embd % config.n_head == 0
        self.n_embd = n_embd
        self.config = config
        self.use_soft_experts = use_soft_experts
        self.nb_non_shared_soft_experts = config.nb_soft_experts - config.nb_shared_experts
        # key, query, value projections for all heads, but in a batch
        if not use_soft_experts:
          self.csa_to_qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        else:
          if self.config.use_same_kv_for_soft_experts:
            self.csa_to_q = nn.Linear(self.n_embd, self.nb_non_shared_soft_experts * self.n_embd, bias=config.bias)
            self.csa_to_kv = nn.Linear(self.n_embd, 2 * self.n_embd, bias=config.bias)
          else:
            self.csa_to_qkv = nn.Linear(self.n_embd, 3 * (self.n_embd * self.nb_non_shared_soft_experts), bias=config.bias)

        # output projection
        self.csa_proj_to_embd = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.ln_qk = RMSNorm(self.n_embd) if config.normalization == 'rmsnorm' else LayerNorm(self.n_embd, bias=config.bias)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            if not use_soft_experts:
              self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                          .view(1, 1, config.block_size, config.block_size))
            else:
              self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                          .view(1, 1, 1, config.block_size, config.block_size))


    def compute_qkv(self, x):
        B, T, C = x.size()
        if not self.use_soft_experts:
          q, k, v  = self.csa_to_qkv(x).split(self.n_embd, dim=-1)
          # q, k layer norm before attention
          q, k = self.ln_qk(q), self.ln_qk(k)
          q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
          k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
          v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        else:
          if self.config.use_same_kv_for_soft_experts:
            q = self.csa_to_q(x).view(B, T, self.nb_non_shared_soft_experts, self.n_embd).transpose(1, 2) # B, nse, T, n_embd
            k, v  = self.csa_to_kv(x).split(self.n_embd, dim=-1) # B, T, n_embd
            # q, k layer norm before attention
            q, k = self.ln_qk(q), self.ln_qk(k)
            q = q.view(B, self.nb_non_shared_soft_experts, T, self.n_head, self.n_embd // self.n_head).transpose(2, 3) # B, nse, nh, T, hs
            k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2).unsqueeze(1) # (B, 1, nh, T, hs)
            v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2).unsqueeze(1) # (B, 1, nh, T, hs)
          else:
            q, k, v = self.csa_to_qkv(x).split(self.nb_non_shared_soft_experts * self.n_embd, dim=-1) # B, T, nse * n_embd
            q = q.view(B, T, self.nb_non_shared_soft_experts, self.n_embd).transpose(1, 2) # B, nse, T, n_embd
            k = k.view(B, T, self.nb_non_shared_soft_experts, self.n_embd).transpose(1, 2) # B, nse, T, n_embd
            v = v.view(B, T, self.nb_non_shared_soft_experts, self.n_embd).transpose(1, 2) # B, nse, T, n_embd
            # q, k layer norm before attention
            q, k = self.ln_qk(q), self.ln_qk(k)
            q = q.view(B, self.nb_non_shared_soft_experts, T, self.n_head, self.n_embd // self.n_head).transpose(2, 3) # B, nse, nh, T, hs
            k = k.view(B, self.nb_non_shared_soft_experts, T, self.n_head, self.n_embd // self.n_head).transpose(2, 3) # B, nse, nh, T, hs
            v = v.view(B, self.nb_non_shared_soft_experts, T, self.n_head, self.n_embd // self.n_head).transpose(2, 3) # B, nse, nh, T, hs
        return q, k, v



    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length = ng*ls, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        assert C % self.n_head == 0, f"n_embd ({C}) must be a multiple of n_head ({self.n_head})"
        # don't need to send a copy bc args aren't modified
        q, k, v = self.compute_qkv(x)


        # (B, nh, T, T)
        if not self.flash:
          bias_mask = self.bias[:,:,:T,:T] == 0 if not self.use_soft_experts else self.bias[:,:,:,:T,:T] == 0
        else:
          bias_mask = None
        # don't need to send a copy of q, k , v
        y = compute_attn(q, k, v, self.attn_dropout, self.flash, self.config, self.training, bias_mask)
        # (B, ls, C)
        if not self.use_soft_experts:
          y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        else:
          # (B, nse, ls, C)
          y = y.transpose(2, 3).contiguous().view(B, self.nb_non_shared_soft_experts, T, C)

        # output projection
        return self.resid_dropout(self.csa_proj_to_embd(y)) # B, T=ng*ls, n_embd


class MLP(nn.Module):

    def __init__(self, config, n_embd, csa_type = 'global', use_soft_experts = False):
        super().__init__()
        # double hdim because gating in swiglu activation
        self.config = config
        self.n_embd = n_embd
        self.csa_type = csa_type
        self.mlp_inner_proj    = nn.Linear(self.n_embd, 2 * (4 * self.n_embd), bias=config.bias) if config.activation =='swiglu' else nn.Linear(self.n_embd, 4 * self.n_embd, bias=config.bias)
        self.activation    = SwiGLU() if  config.activation =='swiglu' else nn.GELU()
        self.use_soft_experts = use_soft_experts
        self.nb_non_shared_soft_experts = config.nb_soft_experts - config.nb_shared_experts
        if not use_soft_experts:
          self.mlp_out_proj  = nn.Linear(4 * self.n_embd, self.n_embd, bias=config.bias)
        else:
          assert self.nb_non_shared_soft_experts != 0, "should have at least one non shared expert to consider a mlp soft expert case"
          self.soft_experts_out_dim = int(self.config.coeff_soft_expert_mlp_dim * self.n_embd // self.config.nb_soft_experts)
          self.non_shared_soft_experts = nn.ParameterList([nn.Parameter(torch.randn(4 * self.n_embd, self.soft_experts_out_dim),
                                                      requires_grad=True) for _ in range(self.nb_non_shared_soft_experts)]) # enable parallel computation with stack # don't bother with bias
          if config.nb_shared_experts != 0:
            self.shared_soft_experts = nn.ParameterList([nn.Parameter(torch.randn(4 * self.n_embd, self.soft_experts_out_dim),
                                                    requires_grad=True) for _ in range(config.nb_shared_experts)])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
      x = self.activation(self.mlp_inner_proj(x))
      if not self.use_soft_experts:
        return self.dropout(self.mlp_out_proj(x))

      if (self.csa_type == 'local'):
        B, ng, nse, T, C = x.size()
        if self.config.nb_shared_experts != 0:
          y = (
              self.dropout(torch.einsum('bgntc,ncd->bgntd', x, torch.stack(list(self.non_shared_soft_experts)))) # B, ng, nse, T, soft_experts_out_dim
              +
              self.dropout(
                  torch.einsum('bgmntc,mncd->bgmntd', 
                               x.unsqueeze(2).expand(-1, -1, self.config.nb_shared_experts, -1, -1, -1),
                               torch.stack(list(self.shared_soft_experts)).unsqueeze(1).expand(-1, nse, -1, -1)
                  ).sum(dim=2)
              )
          )
        else:
          y = self.dropout(torch.einsum('bgntc,ncd->bgntd', x, torch.stack(list(self.non_shared_soft_experts)))) # B, ng, nse, T, soft_experts_out_dim
        return y.contiguous()

      B, nse, T, C = x.size()
      if self.config.nb_shared_experts != 0:
          y = (
                self.dropout(torch.einsum('bntc,ncd->bntd', x, torch.stack(list(self.non_shared_soft_experts)))) # B, nse, T, soft_experts_out_dim
                +
                self.dropout(
                    torch.einsum('bmntc,mncd->bmntd', x.unsqueeze(1).expand(-1, self.config.nb_shared_experts, -1, -1, -1), 
                                 torch.stack(list(self.shared_soft_experts)).unsqueeze(1).expand(-1, nse, -1, -1)
                    ).sum(dim=1)
                ) 
          )
      else:
          y = self.dropout(torch.einsum('bntc,ncd->bntd', x, torch.stack(list(self.non_shared_soft_experts)))) # B, nse, T, soft_experts_out_dim
      return y.contiguous()


class BlockCausalSA(nn.Module):

    def __init__(self, config, in_n_embd, out_n_embd, csa_type = 'global', use_soft_experts = False):
        super().__init__()
        assert (csa_type == 'local' or csa_type =='global'), "Only 'local' or 'global' attn type "
        self.config = config
        self.csa_type = csa_type
        self.in_n_embd = in_n_embd
        self.out_n_embd = out_n_embd
        self.use_soft_experts = use_soft_experts
        self.nb_non_shared_soft_experts = config.nb_soft_experts - config.nb_shared_experts
        self.ln_1 = RMSNorm(self.in_n_embd) if config.normalization == 'rmsnorm' else LayerNorm(self.in_n_embd, bias=config.bias)
        self.block_csa_dropout = nn.Dropout(config.dropout)
        if (csa_type == 'local'):
          self.attn = LocalCausalSelfAttention(config, self.in_n_embd, self.out_n_embd, self.use_soft_experts)
          self.res_proj_up_dim = nn.Linear(self.in_n_embd, self.out_n_embd, bias=config.bias) if (self.in_n_embd != self.out_n_embd) else None
        else:
          assert self.in_n_embd == self.out_n_embd, "should have the same embd dim in and out for global csa block"
          self.attn = GlobalCausalSelfAttention(config, self.out_n_embd, self.use_soft_experts)
        self.ln_2 = RMSNorm(self.out_n_embd) if config.normalization == 'rmsnorm' else LayerNorm(self.out_n_embd, bias=config.bias)
        self.mlp = MLP(config, self.out_n_embd, csa_type, self.use_soft_experts)
        if self.use_soft_experts:
          self.soft_experts_out_dim = int(self.config.coeff_soft_expert_mlp_dim * self.out_n_embd // self.config.nb_soft_experts)
          self.soft_experts_proj = nn.Linear(self.soft_experts_out_dim, self.out_n_embd, bias=config.bias)

    def forward(self, x, local_mem_tokens = None, mem_tokens = None):
        assert (mem_tokens != None and self.up_dim) or mem_tokens == None, "when define mem token should activate the up dim"

        if self.csa_type == 'local':
          if local_mem_tokens != None:
            local_mem_tokens = self.ln_1(local_mem_tokens)
          local_mem_tokens, local_attn_out = self.attn(self.ln_1(x), local_mem_tokens)
          if local_mem_tokens != None:
            local_mem_tokens = self.mlp(self.ln_2(local_mem_tokens))
          mem_tokens = self.res_proj_up_dim(mem_tokens) if mem_tokens != None else mem_tokens

          if self.use_soft_experts:
            x = x.unsqueeze(2).expand(-1, -1, self.config.nb_soft_experts, -1, -1)

          x = self.block_csa_dropout(self.res_proj_up_dim(x)) + local_attn_out if (self.in_n_embd != self.out_n_embd) else (x + local_attn_out)
          if not self.use_soft_experts:
            return mem_tokens, local_mem_tokens, x + self.mlp(self.ln_2(x))
          y = x + self.block_csa_dropout(self.soft_experts_proj(self.mlp(self.ln_2(x))))
          return mem_tokens, local_mem_tokens, (y.softmax(dim=2) * y).sum(dim=2)

        local_attn_out = self.attn(x)
        if self.use_soft_experts:
            x = x.unsqueeze(1).expand(-1, self.nb_non_shared_soft_experts, -1, -1)
        x = x + local_attn_out

        if not self.use_soft_experts:
          return x + self.mlp(self.ln_2(x))
        y = x + self.block_csa_dropout(self.soft_experts_proj(self.mlp(self.ln_2(x))))
        return (y.softmax(dim=1) * y).sum(dim=1) # B, nse, ls, n_embd

class TransformerEncoder(nn.Module):

    def __init__(self, config, idx_n_embd):
        super().__init__()
        assert config.n_block_transformer_enc >= 1, "should have at least one layer in the transformer encoder"
        self.config = config
        self.in_n_embd = config.n_embd[idx_n_embd]
        self.out_n_embd = config.n_embd[idx_n_embd + 1]
        self.latent_size = config.latent_size[idx_n_embd]
        use_soft_experts_in_enc =  (config.nb_soft_experts != 0) and ('enc' in self.config.apply_soft_experts_in_blocks)
        self.up_dim_layer = BlockCausalSA(config, self.in_n_embd, self.out_n_embd, csa_type = 'local', use_soft_experts = use_soft_experts_in_enc) # should be global
        if config.n_block_transformer_enc > 1:
          self.attn_transfo_enc_blocks = nn.ModuleList([BlockCausalSA(config, self.out_n_embd, self.out_n_embd, csa_type = 'local',
                                                                      use_soft_experts = use_soft_experts_in_enc and (idx + 1) % config.apply_soft_experts_every == 0) for idx in range(config.n_block_transformer_enc - 1)])
        else:
          self.attn_transfo_enc_blocks = None

    def forward(self, x, local_mem_tokens = None, mem_tokens = None):
        B, ng, lg, C = x.size() # batch size, n_groups, length group, embedding dimensionality (n_embd)
        mem_tokens, local_mem_tokens, x = self.up_dim_layer(x, local_mem_tokens, mem_tokens) # send global mem tokens only when up sample
        if self.config.n_block_transformer_enc > 1:
          for transfo_enc_block in self.attn_transfo_enc_blocks: # act on group, only use local mem tok
            _, local_mem_tokens, x = transfo_enc_block(x, local_mem_tokens, None) # (B, ng, ls, out_n_embd)
        return mem_tokens, local_mem_tokens, x[:, :, -self.latent_size:, :] # (B, ng, ls, out_n_embd)

def compute_local_global_attn(x, mem_tokens, local_csa_mem_tokens, attn_local_blocks, attn_global_blocks, config):
    B, ng, T, C = x.size()
    for local_csa_block, global_csa_block in zip(attn_local_blocks, attn_global_blocks):
      # mem token not sent for regular local attn, only when enc
      _, local_csa_mem_tokens, x = local_csa_block(x, local_csa_mem_tokens, None) # B, ng, ls, out_n_embd
      if config.interleave_local_global_csa:
        x = x.view(B, -1, C) # B, ng * ls, out_n_embd
        if config.nb_mem_token != 0:
          x = torch.cat((mem_tokens, x), dim=1) # B, ng * ls, out_n_embd with ng * ls = ng * ls + nb_mem_tokens
        x = global_csa_block(x) # B, ng * ls, out_n_embd
        # drop mem token when doing local attn in order to not break the causality by leaking info, groups are in the time dim, they cannot share the same mem token
        if config.nb_mem_token != 0:
          mem_tokens, x = x[:, :config.nb_mem_token, :], x[:,config.nb_mem_token:,:]
      x = x.view(B, ng, T, C) # in case we interleaved previously make sure we go back to the right shape for local attn

    if not config.interleave_local_global_csa:
      x = x.view(B, -1, C) # B, ng * ls, out_n_embd
      if config.nb_mem_token != 0:
        x = torch.cat((mem_tokens, x), dim=1) # B, ng * ls, out_n_embd with ng * ls = ng * ls + nb_mem_tokens
      for global_csa_block in attn_global_blocks:
        x = global_csa_block(x) # B, ng * ls, out_n_embd
    return local_csa_mem_tokens, x


class EncoderBlock(nn.Module):

    def __init__(self, config, idx_n_embd):
        super().__init__()
        self.in_n_embd = config.n_embd[idx_n_embd]
        self.out_n_embd = config.n_embd[idx_n_embd + 1]
        self.latent_size = config.latent_size[idx_n_embd]
        self.config = config
        self.enc = PerceiverEncoder(config, idx_n_embd) if config.enc_type == 'perceiver' else TransformerEncoder(config, idx_n_embd)
        self.enc_dropout = nn.Dropout(config.enc_dropout)
        self.attn_local_blocks = nn.ModuleList([BlockCausalSA(config, self.out_n_embd, self.out_n_embd, 'local') for _ in range(config.n_block_local_global_causal_sa)])
        self.attn_global_blocks = nn.ModuleList([BlockCausalSA(config, self.out_n_embd, self.out_n_embd, 'global') for _ in range(config.n_block_local_global_causal_sa)])

    def forward(self, x, pos_embd_layer = None, local_csa_mem_tokens = None):
        if self.config.nb_mem_token != 0:
          mem_tokens, x = x[:, :self.config.nb_mem_token, :], x[:,self.config.nb_mem_token:,:]
        else:
          assert local_csa_mem_tokens == None, "local and global mem tokens should be None at the same time"
          mem_tokens = None
        B, T, C = x.size()
        assert T % self.config.n_groups == 0, "latent length must be a multiple of nb of groups"
        assert T >= self.latent_size, "in order to reduce the ubput length we should have the initial seq length greater than the latent size"
        # assert min(self.config.latent_size) >= self.config.n_groups, "should have at least one token per group"
        x = x.view(B, self.config.n_groups, T // self.config.n_groups, C)

        mem_tokens, local_csa_mem_tokens, x = self.enc(x, local_csa_mem_tokens, mem_tokens) # (B, nb_mem_tokens, out_n_embd), (B, ng, ls, out_n_embd)
        # (B, ng, ls, out_n_embd)
        x = self.enc_dropout(x) + pos_embd_layer(torch.arange(0, x.shape[-2], dtype=torch.long, device=x.device)) if pos_embd_layer != None else self.enc_dropout(x)

        return compute_local_global_attn(x, mem_tokens, local_csa_mem_tokens, self.attn_local_blocks, self.attn_global_blocks, self.config)

class DecoderBlock(nn.Module):

    def __init__(self, config, idx_n_embd):
        super().__init__()
        self.idx_n_embd = idx_n_embd
        self.in_n_embd = config.n_embd[config.n_layer_enc_dec - idx_n_embd] # n_layer = len(n_embd) - 1, so in_n_embd goes from n_layer to 1
        self.out_n_embd = config.n_embd[config.n_layer_enc_dec - idx_n_embd - 1] # out_n_embd goes from n_layer - 1 to 0
        self.latent_size = config.latent_size[idx_n_embd]
        self.config = config
        self.dec_proj = nn.Linear(self.in_n_embd, self.out_n_embd, bias=config.bias)
        self.dec_dropout = nn.Dropout(config.dec_dropout)
        self.attn_local_blocks = nn.ModuleList([BlockCausalSA(config, self.out_n_embd, self.out_n_embd, 'local') for _ in range(config.n_block_local_global_causal_sa)])
        self.attn_global_blocks = nn.ModuleList([BlockCausalSA(config, self.out_n_embd, self.out_n_embd, 'global') for _ in range(config.n_block_local_global_causal_sa)])
        self.in_T = self.config.latent_size[config.n_layer_enc_dec - 1 - idx_n_embd]
        self.out_T = self.config.latent_size[config.n_layer_enc_dec - 1 - idx_n_embd - 1] if idx_n_embd < config.n_layer_enc_dec - 1 else self.config.block_size
        if self.out_T - self.in_T != 0:
          self.first_group_tokens_for_upsampling = nn.Parameter(torch.randn(self.out_T - self.in_T, self.in_n_embd),
                                                              requires_grad = True) # for now assume ok to have same param for different exapmles in the batch
        else:
          self.first_group_tokens_for_upsampling = nn.Parameter(torch.randn(self.in_T, self.in_n_embd),
                                                              requires_grad = True)
        if config.up_sampling_type == 'attn':
          self.dec_to_q = nn.Linear(self.out_n_embd, self.out_n_embd, bias=config.bias)
          self.dec_to_k = nn.Linear(self.out_n_embd, self.out_n_embd, bias=config.bias)
          self.dec_to_v = nn.Linear(self.out_n_embd, self.out_n_embd, bias=config.bias)
          self.n_head = config.n_head
          self.ln_qk = RMSNorm(self.out_n_embd) if config.normalization == 'rmsnorm' else LayerNorm(self.out_n_embd, bias=config.bias)
          self.attn_dropout = nn.Dropout(config.dropout)
          # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
          self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
          if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, 1, config.block_size, config.block_size))

    def compute_qkv(self, in_proj_to_q, in_proj_to_k, in_proj_to_v):
      B, ng, out_T, out_n_embd = in_proj_to_q.size()
      q  = self.dec_to_q(in_proj_to_q) # B, ng, out_T, out_n_embd # local mem tok will also be up dim here if any
      k  = self.dec_to_k(in_proj_to_k)
      v  = self.dec_to_v(in_proj_to_v)
      # q, k layer norm before attention
      q, k = self.ln_qk(q), self.ln_qk(k)
      q = q.view(B, ng, out_T, self.n_head, self.out_n_embd // self.n_head).transpose(2, 3) # (B, ng, nh, ls, hs)
      k = k.view(B, ng, out_T, self.n_head, self.out_n_embd // self.n_head).transpose(2, 3) # (B, ng, nh, ls, hs)
      v = v.view(B, ng, out_T, self.n_head, self.out_n_embd // self.n_head).transpose(2, 3) # (B, ng, nh, ls, hs)
      return q, k, v


    def forward(self, x, input_skip, pos_embd_layer = None, local_csa_mem_tokens = None, initial_length = None):

        if self.config.nb_mem_token != 0:
          mem_tokens, x = x[:, :self.config.nb_mem_token, :], x[:,self.config.nb_mem_token:,:] # separate global mem token from input

        else:
          assert local_csa_mem_tokens == None, "local and global mem tokens should be None at the same time"
          mem_tokens = None

        device = x.device
        B, T, C = x.size()
        assert T % self.config.n_groups == 0, "latent length must be a multiple of nb of groups"
        assert T // self.config.n_groups <= self.out_T, f"in order to increase the input length we should have the initial seq length ({T // self.config.n_groups}) greater than the output length {self.out_T}"
        # assert min(self.config.latent_size) >= self.config.n_groups, "should have at least one token per group"
        x = x.view(B, self.config.n_groups, T // self.config.n_groups, C)
        if self.idx_n_embd == self.config.n_layer_enc_dec - 1:
          assert (input_skip.shape[1] % self.config.n_groups == 0), "input length must be a multiple of the nb of groups"
          self.out_T = input_skip.shape[1] // self.config.n_groups # self.out_T & self.in_T are per group, we want to reconstruct the input length at the end

        nb_tok_to_add = self.out_T  - self.in_T
        segment_size = (nb_tok_to_add) // (self.in_T) + 1

        if nb_tok_to_add != 0:
          # upsampling by taking in the previous group and preserve causality
          all_prefixes = torch.cat([self.first_group_tokens_for_upsampling.expand(B, 1, -1, C).to(device)[:, :, -(nb_tok_to_add):, :],
                                    x[:, :-1].repeat(1, 1, segment_size, 1)[:, :, -(nb_tok_to_add):, :]], dim=1).contiguous() # B, ng, nb_tok, C
          x = self.dec_dropout(self.dec_proj(torch.cat([all_prefixes, x], dim=2))) # B, ng, (nb_tok + in_T) = out_T, out_n_embd
        else:
          x = self.dec_dropout(self.dec_proj(x)) # B, ng, in_T, out_n_embd

        if pos_embd_layer != None:
          x += pos_embd_layer(torch.arange(0, x.shape[-2], dtype=torch.long, device=x.device))


        if self.config.up_sampling_type == 'attn':
          # B, ng, nh, out_T, out_T
          bias_mask = self.bias[:, :, :, :self.out_T, :self.out_T] == 0 if not self.flash else None
          # don't need to make a copy/clone here bc args aren't modified
          q, k, v = self.compute_qkv(in_proj_to_q = input_skip.view(x.size()) + x, in_proj_to_k = x, in_proj_to_v = x)
          if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            scale = (1.0 / k.size(-1)) if self.config.muP else (1.0 / math.sqrt(k.size(-1)))
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout \
                                                                 if self.training else 0, is_causal=True, scale=scale)
          else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1)) if self.config.muP else (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(bias_mask, -torch.finfo(q.dtype).max)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
          x = input_skip.view(x.size()) + x + y.transpose(2, 3).contiguous().view(x.size())

        return compute_local_global_attn(x, mem_tokens, local_csa_mem_tokens, self.attn_local_blocks, self.attn_global_blocks, self.config)

@dataclass
class AnyModalMirasolConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer_enc_dec: int = 3 # nb block enc/dec
    n_head: int = 8
    n_embd: list[int] = field(default_factory=lambda: [256, 1024, 2048, 2048]) # input embd first and then latent embd dim
    dropout: float = 0.0
    enc_dropout: float = 0.0
    enc_type: str = 'perceiver' # perceiver, tranfo_enc
    dec_dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    activation: str = 'swiglu' # gelu or swiglu
    normalization: str = 'rmsnorm' # layernorm or rmsnorm
    noisy_embd_alpha: float = 10.0 # 0 if you don't want to use noisy embedding # https://openreview.net/pdf?id=0bMmZ3fkCk
    nb_mem_token: int = 0 # nb memory / sink token # https://arxiv.org/pdf/2309.17453
    muP: bool = True # hyper-param transfer muP # https://arxiv.org/pdf/2203.03466
    apply_gradient_checkpointing: bool = False # trade compute for memory
    gradient_checkpoint_sequential_factor: int = 2
    pos_embd: str = 'simple' # simple, no
    n_groups: int = 8
    latent_size: list[int] = field(default_factory=lambda: [64, 32, 16])
    n_block_local_global_causal_sa: int = 2
    n_block_latent_attn: int = 2
    n_block_transformer_enc: int = 2
    interleave_local_global_csa: bool = False # apply the fused block csa after each local csa or wait after all local csa apply to each group and then do all the global attn
    lernable_skip_hiearchy: bool = True
    latent_causal_loss_type: str = 'cos_similarity' # None, 'cos_similarity' # https://arxiv.org/pdf/2311.05698
    nb_soft_experts: int = 2
    nb_shared_experts: int = 1 # add to the number of soft experts
    use_same_kv_for_soft_experts: bool = True
    apply_soft_experts_every: int = 1
    coeff_soft_expert_mlp_dim: int = 1 # each soft expert will have coeff_soft_expert_mlp_dim * n_embd / nb_soft_experts as output dim
    apply_soft_experts_in_blocks: list[str] = field(default_factory=lambda: ['latent']) # enc, latent
    init_weight: str = 'nanogpt' # 'nanogpt', 'spike_no_more' # overrides if muP
    use_dense_former_in_latent: bool = True # https://arxiv.org/pdf/2402.02622
    use_pos_embd_in_enc_dec: bool = True
    up_sampling_type: str = 'attn' # 'attn' or 'repeat'

def init_with_muP(pn, p, config):
  special_weights = ['first_group_tokens_for_upsampling', 'lm_head.weight', 'shared_soft_experts', 'pos_embd_layers', 'wte', 'mem_token']
  total_n_layer = 2 * config.n_layer_enc_dec * (2 * config.n_block_local_global_causal_sa) + config.n_block_latent_attn
  coeff_std = math.sqrt(2 / 5) / math.sqrt(2 * total_n_layer)
  if (p.dim() == 2) and not any(sub in pn for sub in special_weights):
    fan_in = p.shape[1] # pytorch default linear shape: (out_features, in_features)
    torch.nn.init.normal_(p, mean=0.0, std=coeff_std/(math.sqrt(fan_in)))

  elif 'shared_soft_experts' in pn: # shared and non_shared
    fan_in = p.shape[0] # (in_features, out_features)
    torch.nn.init.normal_(p, mean=0.0, std=coeff_std/(math.sqrt(fan_in)))

  elif 'lm_head.weight' in pn:
    torch.nn.init.zeros_(p)

def init_with_spike_no_more(pn, p, config):
  # spike no more init: std=sigma/sqrt(2N) with sigma = sqrt(2/5d) and N = nb layers for W2, W0 # https://arxiv.org/pdf/2312.16903
  total_n_layer = 2 * config.n_layer_enc_dec * (2 * config.n_block_local_global_causal_sa) + config.n_block_latent_attn
  coeff_std = math.sqrt(2 / 5) / math.sqrt(2 * total_n_layer)
  if any(subname in pn for subname in ['proj_to_embd', 'out_proj']):
    fan_in = p.shape[1] # pytorch default linear shape: (out_features, in_features)
    torch.nn.init.normal_(p, mean=0.0, std=coeff_std/(math.sqrt(fan_in)))
  elif 'shared_soft_experts' in pn: # shared and non_shared
    fan_in = p.shape[0] # (in_features, out_features)
    torch.nn.init.normal_(p, mean=0.0, std=coeff_std/(math.sqrt(fan_in)))

def init_with_nano_gpt(pn, p, config):
  total_n_layer = 2 * config.n_layer_enc_dec * (2 * config.n_block_local_global_causal_sa) + config.n_block_latent_attn
  if any(subname in pn for subname in ['proj_to_embd', 'out_proj', 'shared_soft_experts']):
    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * total_n_layer))


def init_remaining_params(model_params, config):
  for pn, p in model_params:
    if config.muP:
      init_with_muP(pn, p, config)

    elif config.init_weight == 'spike_no_more':
      init_with_spike_no_more(pn, p, config)

    else: # keep nanogpt init as defautlt
      init_with_nano_gpt(pn, p, config)

class AnyModalMirasol(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert len(config.n_embd) == len(config.latent_size) + 1 and len(config.latent_size) == config.n_layer_enc_dec, "Should have one more n_embd than latent size elms"
        assert config.apply_soft_experts_every >=1, "apply_soft_experts_every should be non-zero"
        self.config = config
        print(config)

        if config.use_dense_former_in_latent:
          self.dense_former_skip_buffer = None
          # pytorch doesn't allow for in-place op in leaf variable with requires_grad = True
          coeffs = torch.zeros(self.config.n_block_latent_attn, self.config.n_block_latent_attn + 1)
          for i in range(self.config.n_block_latent_attn): # init with vanilla transformer connections
            coeffs[i][i+1] = 1.0
          self.dense_former_coeffs = nn.Parameter(coeffs.clone(),
                           requires_grad=True)
          coeffs = None

        self.input_skip_buffer = []

        # if self.config.init_weight == 'spike_no_more':
        #   self.ln_embd = RMSNorm(self.config.n_embd[0]) if config.normalization == 'rmsnorm' else LayerNorm(self.config.n_embd[0], bias=config.bias)

        if self.config.pos_embd == 'simple':
          pos_embd_layers = [nn.Embedding(self.config.block_size, self.config.n_embd[0])]
          if self.config.use_pos_embd_in_enc_dec: # instantiate it here to reuse param between enc and dec
            pos_embd_layers += [nn.Embedding(self.config.latent_size[idx], self.config.n_embd[idx+1]) for idx in range(self.config.n_layer_enc_dec)]
          self.pos_embd_layers = nn.ModuleList(pos_embd_layers)

        use_soft_experts_in_latent =  (config.nb_soft_experts != 0) and ('latent' in self.config.apply_soft_experts_in_blocks)
        self.any_modal_mirasol = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd[0]),
            drop = nn.Dropout(config.dropout),
            enc_h = nn.ModuleList([EncoderBlock(config, idx) for idx in range(config.n_layer_enc_dec)]),
            latent_attn_h = nn.ModuleList([BlockCausalSA(config, config.n_embd[-1], config.n_embd[-1],
                                                         'global', use_soft_experts_in_latent and (idx + 1) % config.apply_soft_experts_every == 0) for idx in range(config.n_block_latent_attn)]),
            dec_h = nn.ModuleList([DecoderBlock(config, idx) for idx in range(config.n_layer_enc_dec)]),
            # ln_f = RMSNorm(config.n_embd[0]) if config.normalization == 'rmsnorm' else LayerNorm(config.n_embd[0], bias=config.bias),
        ))

        if self.config.latent_causal_loss_type is not None:
          assert self.config.latent_causal_loss_type in {'cos_similarity'}, f"please select a valid latent loss, current input {self.config.latent_causal_loss_type}, possible choice: 'cos_similarity'"
          self.latent_causal_loss = LatentCausalReconstructionLoss()
          self.latent_prediction_head = nn.Linear(self.config.n_embd[-1], self.config.n_embd[-1], bias = True)

        self.lm_head = nn.Linear(config.n_embd[0], config.vocab_size, bias = self.config.bias)
        self.res_lerp = nn.Parameter(torch.randn(self.config.n_layer_enc_dec), # n_layer res connection between skip and upsample # torch.randn(self.config.n_layer)
                           requires_grad=True) if self.config.lernable_skip_hiearchy else None # make sure  is learnable

        if self.config.nb_mem_token != 0:
          self.mem_token = nn.Parameter(torch.randn(self.config.nb_mem_token, self.config.n_embd[0]), # [batch_size, nb_memory_token, n_embd]
                           requires_grad=True) # make sure the embedding is learnable
          self.local_csa_mem_tokens = nn.Parameter(torch.randn(self.config.n_groups * self.config.nb_mem_token, self.config.n_embd[0]), # [batch_size, nb_memory_token, n_embd]
                           requires_grad=True) # make sure the embedding is learnable

        self.total_n_layer = 2 * config.n_layer_enc_dec * (2 * config.n_block_local_global_causal_sa) + config.n_block_latent_attn
          
        # init all weights
        self.apply(self._init_weights)
        init_remaining_params(self.named_parameters(), config)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position esmmstmbeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= (self.pos_embd_layers[0].weight.numel() + self.config.nb_mem_token) if self.config.nb_mem_token != 0 else self.pos_embd_layers[0].weight.numel()
            if self.config.use_pos_embd_in_enc_dec:
              n_params -= sum(layer.weight.numel() for layer in self.pos_embd_layers[1:])

        return n_params

    def _init_weights(self, module):
        """
        default init
        """
        if isinstance(module, nn.Linear):
          if self.config.muP: # must deal with it first
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(self.config.n_embd[0])) # muP
          elif self.config.init_weight == 'spike_no_more':
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          else:  # nanoGPT as default
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

          if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
          if self.config.muP: 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if self.config.init_weight == 'spike_no_more':
          # torch.nn.init.normal_(module.weight, mean=0.0, std=1) # spike no more paper https://arxiv.org/pdf/2312.16903
            torch.nn.init.normal_(module.weight, mean=0.0, std= math.sqrt(2 / (5 * self.config.n_embd[0]))) # account for tok embd and all pos embd
          else: # nanoGPT as default
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, global_coeff_latent_causal_loss = 0.2, step_size_eval = 1):
        device = idx.device
        b, t = idx.size()
        length_padding = 0
        if t % self.config.n_groups != 0: # pad the last group if needed # should have a special pad
          length_padding = self.config.n_groups - t % self.config.n_groups
          t += length_padding
          idx = torch.cat([idx, torch.zeros(b, length_padding, dtype = idx.dtype).to(device)], dim = -1)
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"


        # forward the GPT model itself
        if self.config.init_weight == 'spike_no_more':
          tok_emb = self.any_modal_mirasol.wte(idx) * self.config.n_embd[0] # token embeddings of shape (b, t, n_embd) # scaled embed
          # tok_emb = self.ln_embd(tok_emb) # for now ignore the embd ln
        else:
          tok_emb = self.any_modal_mirasol.wte(idx) # token embeddings of shape

        if (self.config.noisy_embd_alpha != 0.0):
          eps = 2 * torch.rand(1) - 1
          tok_emb += ((self.config.noisy_embd_alpha / math.sqrt(tok_emb.size(-2) * tok_emb.size(-1))) * eps).to(device)

        if self.config.pos_embd == 'no' or self.config.pos_embd == 'rope':
          pos_emb = torch.zeros(t, self.config.n_embd[0], dtype=torch.long, device=device) # (t, n_embd)
        else:
          pos_emb = self.pos_embd_layers[0](torch.arange(0, t, dtype=torch.long, device=device)) # position embeddings of shape (t, n_embd)

        # add memory/sink token after the embedding if requires
        if self.config.nb_mem_token != 0:
          mem_token = self.mem_token.expand(b, -1, -1).to(device)
          local_csa_mem_tokens = self.local_csa_mem_tokens.expand(b, -1, -1)
        else:
          local_csa_mem_tokens = None
        x = self.any_modal_mirasol.drop(torch.cat((mem_token, tok_emb + pos_emb), dim=1)) if self.config.nb_mem_token != 0 else self.any_modal_mirasol.drop(tok_emb + pos_emb)
        tok_emb, pos_emb = None, None # free memory as soon as possible
        self.input_skip_buffer.append(x.clone()) # register the first skip conection

        # encoder
        for i, block in enumerate(self.any_modal_mirasol.enc_h):
          pos_embd_layer = self.pos_embd_layers[i+1] if self.config.use_pos_embd_in_enc_dec else None
          local_csa_mem_tokens, x = block(x, pos_embd_layer = pos_embd_layer, local_csa_mem_tokens = local_csa_mem_tokens)
          if i != self.config.n_layer_enc_dec - 1: #
            self.input_skip_buffer.append(x)

        if self.config.use_dense_former_in_latent:
          self.dense_former_skip_buffer = x.clone().view(1, b, -1, self.config.n_embd[-1])


        # main processing
        # apply gradient checkpointing if set # only the ckpt_sequential api works. why?
        if self.config.apply_gradient_checkpointing:
          assert not self.config.use_dense_former_in_latent, "can't currently combine gradient checkpointing and denseformer in the latent"
          x = checkpoint_sequential(self.any_modal_mirasol.latent_attn_h,
                                    self.config.n_block_latent_attn // self.config.gradient_checkpoint_sequential_factor,
                                    x, use_reentrant = False) # B, ng * ls, out_n_embd
        else:
          for i, latent_csa_block in enumerate(self.any_modal_mirasol.latent_attn_h):
            x = latent_csa_block(x) # B, ng * ls, out_n_embd
            if self.config.use_dense_former_in_latent:
              self.dense_former_skip_buffer = torch.cat([self.dense_former_skip_buffer,
                                                         x.view(1, b, -1, self.config.n_embd[-1])], dim=0)
              x = (self.dense_former_skip_buffer * self.dense_former_coeffs[i][:i+2].view(-1, 1, 1, 1).expand_as(self.dense_former_skip_buffer)).sum(dim=0)

        # latent causal reconstruction
        if self.config.latent_causal_loss_type is not None:
          # x *= (torch.rand_like(x) < 0.0075).type(x.dtype) # random masking
          latent_reconstruction_targets = x[:, 1:, :]
          x = self.latent_prediction_head(x)
          latent_reconstruction_predictions = x[:, :-1, :]


        # decoder
        for i, decodeur_block in enumerate(self.any_modal_mirasol.dec_h):
          pos_embd_layer = self.pos_embd_layers[-i-2] if self.config.use_pos_embd_in_enc_dec else None
          local_csa_mem_tokens, x = decodeur_block(x, input_skip = self.input_skip_buffer[-1], pos_embd_layer = pos_embd_layer,
                                                   local_csa_mem_tokens = local_csa_mem_tokens)
          if self.config.lernable_skip_hiearchy:
            x = self.res_lerp[i] * self.input_skip_buffer.pop() + (1 - self.res_lerp[i]) * x
          else:
            x += self.input_skip_buffer.pop()

        # x = self.any_modal_mirasol.ln_f(x)

        if targets is not None:
            loss = {"total": torch.tensor(0.0), "cross_entropy": torch.tensor(0.0), "latent_causal_modeling": torch.tensor(0.0)}
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) / self.config.n_embd[0] if self.config.muP else self.lm_head(x) # muP

            if length_padding != 0:
              logits = logits[:, :-length_padding, :]

            if self.training:
              # discard memory token if any
              if self.config.nb_mem_token != 0:
                logits = logits[:,self.config.nb_mem_token:,:]
                loss["total"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
              else:
                loss["total"] = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
              logits = logits[:, -step_size_eval:, :] # # can default to step_size=block size if want the full ct eval
              loss["total"] = F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)),
                                   targets[:, -step_size_eval:].contiguous().view(-1), ignore_index=-1)

            loss["cross_entropy"] = loss["total"]

            if self.config.latent_causal_loss_type is not None:
              assert 0.0 <= global_coeff_latent_causal_loss < 1.0, "latent_causal_modeling_loss_coeff should be between 0 and 1"
              loss["latent_causal_modeling"] = self.latent_causal_loss(latent_reconstruction_predictions, latent_reconstruction_targets)
              loss["total"] = (1.0 - global_coeff_latent_causal_loss) * loss["cross_entropy"] + global_coeff_latent_causal_loss * loss["latent_causal_modeling"]
        else:
            if length_padding != 0:
              x = x[:, :-length_padding, :]
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim # only want to predict the char for the last elm of the seq, use [-1] to keep the seq dim when slicing
            logits = logits / self.config.n_embd[0] if self.config.muP else logits # muP # doesn't impact in fact
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.any_modal_mirasol.wpe.weight = nn.Parameter(self.any_modal_mirasol.wpe.weight[:block_size])
        for block in self.any_modal_mirasol.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, use_8_bit_optim, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        print(f"using muP: {self.config.muP}")


        if self.config.muP:
          special_weights = ['pos_embd_layers', 'wte', 'first_group_tokens_for_upsampling', 'shared_soft_experts', 'mem_token']
          linear_decay_fan_in_params = {p.shape[1]: [] for n, p in param_dict.items() if (p.dim() == 2) and not any(sub in n for sub in special_weights)}
          linear_decay_params_names = []
          soft_expert_linear_decay_fan_in_params = {p.shape[0]: [] for n, p in param_dict.items() if 'shared_soft_experts' in n}
          soft_expert_linear_decay_params_names = []
          others_params_decay = []
          others_params_names_decay = []
          for pn,p in param_dict.items():
            if p.dim() >=2: # skip no_decay params
              if p.dim() == 2 and not any(sub in pn for sub in special_weights):
                  linear_decay_fan_in_params[p.shape[1]].append(p)
                  linear_decay_params_names.append(pn)
              elif 'shared_soft_experts' in pn:
                soft_expert_linear_decay_fan_in_params[p.shape[0]].append(p)
                soft_expert_linear_decay_params_names.append(pn)
              else:
                others_params_decay.append(p)
                others_params_names_decay.append(pn)

          linear_decay_params_group = [
              {'params': linear_decay_fan_in_params[fan_in], 'weight_decay': weight_decay, 'muPScale': fan_in, 'lr': learning_rate / fan_in} for fan_in in linear_decay_fan_in_params.keys()
              ]

          soft_expert_linear_decay_params_groups = [
              {'params': soft_expert_linear_decay_fan_in_params[fan_in], 'weight_decay': weight_decay, 'muPScale': fan_in, 'lr': learning_rate / fan_in} for fan_in in soft_expert_linear_decay_fan_in_params.keys()
              ]
        
          others_params_groups =  [
              {'params': others_params_decay, 'weight_decay': weight_decay, 'muPScale': 1.0},
              {'params': nodecay_params, 'weight_decay': 0.0, 'muPScale': 1.0}
          ]
          optim_groups = linear_decay_params_group + soft_expert_linear_decay_params_groups + others_params_groups
          num_decay_params = sum(p.numel() for fan_in in linear_decay_fan_in_params.keys() for p in linear_decay_fan_in_params[fan_in]) + sum(p.numel() for fan_in in soft_expert_linear_decay_fan_in_params.keys() for p in soft_expert_linear_decay_fan_in_params[fan_in]) + sum(p.numel() for p in others_params_decay)
          num_nodecay_params = sum(p.numel() for p in nodecay_params)
          num_tensors_decay_params = sum(len(linear_decay_fan_in_params[fan_in]) for fan_in in linear_decay_fan_in_params.keys()) + sum(len(soft_expert_linear_decay_fan_in_params[fan_in]) for fan_in in soft_expert_linear_decay_fan_in_params.keys()) + len(others_params_decay)
          print("names linear params with decay: ", linear_decay_params_names)
          print("names soft experts mlp linear params with decay: ", soft_expert_linear_decay_params_names)
          print("names others params with decay: ", others_params_names_decay)
          print("test len decay param: ", num_tensors_decay_params == len(decay_params))

          print(f"num decayed parameter tensors: {num_tensors_decay_params}, with {num_decay_params:,} parameters")
          print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        else:
          optim_groups = [
              {'params': decay_params, 'weight_decay': weight_decay},
              {'params': nodecay_params, 'weight_decay': 0.0}
          ]
          num_decay_params = sum(p.numel() for p in decay_params)
          num_nodecay_params = sum(p.numel() for p in nodecay_params)
          print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
          print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        if not(use_8_bit_optim):
          # Create AdamW optimizer and use the fused version if it is available
          fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # Fused AdamW is a faster and more efficient implementation of the AdamW optimizer in PyTorch
          use_fused = fused_available and device_type == 'cuda'
          extra_args = dict(fused=True) if use_fused else dict()
          optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
          print(f"using fused AdamW: {use_fused}")
        else:
          optimizer = AdamW8bit(optim_groups, lr=learning_rate, betas=betas, weight_decay=0.0) # wd set on param group, don't override with their arg

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
