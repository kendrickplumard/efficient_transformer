# ->light-branch: you can think of it as applying a local window transformer to a reduce number of tokens with a smaller n_embd than 
# the vanilla transformer layers in the overall arch. we consider a group size and process tokens in parallel within their group 
# in the csa. Note that for group=1 we fall into the original mod layer if we ignore the difference of n_embd.
# ->heavy-branch: an intuition is that for a really small number of tokens the model finds interesting, we consider multiple
# versions of them and for each version we let tokens communicate with each other in the csa then apply a special mlp to tokens 
# of this version. in other words it can be view as applying multiple transformer instead of one and this only for really interesting 
# tokens. after that we combine the representations with softmax. it should drastically increase the quality of the embedding of 
# these really important tokens with minimal computational cost since it is parallel and they are not a lot. this is like talking 
# transformers^^. they consider higher input n_embd than the vanilla transformer layers in the overall arch.
# eventually we add some mlps that will be share between versions in order to store common information.
# ->mod: for each of these mod branch we select from the input a batch of queries, then we select a batch of keys with potentially 
# different size and pass them as input to their csa (explanations about mod mechanism https://arxiv.org/pdf/2404.02258)
# so, depending on the config and the selection, a token can skip a layer, pass to a light branch and/or pass to a heavy branch.
# ->vanilla transformer layer: used to incorporate long range dependencies. they also benefit from the processing of mod layers. 

import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

from dataclasses import dataclass
from torchao.prototype.low_bit_optim import AdamW8bit
from torch.utils.checkpoint import checkpoint_sequential


class GatedActivation(nn.Module):
  """ Gated activation function """
  def __init__(self, activation):
    super().__init__()
    self.activation = activation

  def forward(self, x):
    x, gate = x.chunk(2, dim=-1)
    if self.activation == 'swiglu':
      return F.silu(gate) * x
    elif self.activation == 'geglu':
      return F.gelu(gate) * x
    else:
      return F.relu(gate) * x


class RMSNorm(torch.nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim)) # # gain param Learnable scaling parameter.

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

def compute_attn(q, k, v, attn_dropout, use_flash_attn, config, training, attn_mask=None):
  if use_flash_attn:
    # efficient attention using Flash Attention CUDA kernels
    scale = (1.0 / k.size(-1)) if config.muP else (1.0 / math.sqrt(k.size(-1)))
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=config.dropout \
                                                                 if training else 0, is_causal=attn_mask is None, scale=scale)
  else:
    # manual implementation of attention
    att = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1)) if config.muP else (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(attn_mask == 0, -torch.finfo(att.dtype).max)
    att = F.softmax(att, dim=-1)
    att = attn_dropout(att)
    y = att @ v
  return y

def pad_seq(seq, config):
  B, T, C = seq.size() # batch size, sequence length (k), embedding dimensionality (n_embd)
  length_padding = 0
  if T % config.n_groups != 0: # pad the last group if needed # should have a special pad token
    length_padding = config.n_groups - T % config.n_groups
    T += length_padding
    seq = torch.cat([seq, torch.zeros((B, length_padding, C), dtype = seq.dtype).to(seq.device)], dim = 1)

  return seq, T, length_padding

class LightCausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = int(self.config.coeff_light_branch_n_embd * self.config.n_embd)
        assert self.n_embd % config.n_head == 0, f'light branch n_embed({self.n_embd}) should be a multiple of n_head({self.config.n_head})'
        # key, query, value projections for all heads, but in a batch
        self.to_q = nn.Linear(self.n_embd, self.n_embd, bias=self.config.bias)
        self.to_kv = nn.Linear(self.n_embd, 2 * self.n_embd, bias=self.config.bias)
        # output projection
        self.proj_to_embd = nn.Linear(self.n_embd, self.n_embd, bias=self.config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_dropout = nn.Dropout(self.config.dropout)
        self.n_head = self.config.n_head
        self.dropout = self.config.dropout
        self.ln_qk = RMSNorm(self.n_embd) if self.config.normalization == 'rmsnorm' else LayerNorm(self.n_embd, bias=self.config.bias)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def rotate_embeddings(self, x):
        x = x.view(*x.shape[:-1], -1, 2).flip(-1)
        x[...,0] *= -1
        return x.flatten(start_dim=-2)

    def apply_rope(self, q, k):
      B, ng, nh, lg, hs = q.size()
      lgkv = k.shape[-2]
      pos = 10000**((-2 * torch.arange(0, hs, 2, device=x.device) - 1)/ hs)
      token_seq = torch.arange(lg, dtype=pos.dtype, device=x.device).unsqueeze(1) @ pos.unsqueeze(0)
      kv_token_seq = torch.arange(lgkv, dtype=pos.dtype, device=x.device).unsqueeze(1) @ pos.unsqueeze(0)
      pos = None
      rotary_embds = torch.cat((token_seq, token_seq), dim=-1)
      token_seq = None
      kv_rotary_embds = torch.cat((kv_token_seq, kv_token_seq), dim=-1)
      kv_token_seq = None
      q = (q * rotary_embds.cos()) + \
        (self.rotate_embeddings(q) * rotary_embds.sin())
      rotary_embds = None
      k = (k * kv_rotary_embds.cos()) + \
        (self.rotate_embeddings(k) * kv_rotary_embds.sin())
      return q, k


    def forward(self, query_x, key_value_x, causal_mask):

        B, T, C = query_x.size() # batch size, sequence length (k), embedding dimensionality (n_embd)
        Tkv = key_value_x.shape[1]
        assert (B==key_value_x.shape[0] and C==key_value_x.shape[-1]), f"query_value_x and key_value_x should have the same batch({B}!={key_value_x.shape[0]}) and n_embd({C}!={key_value_x.shape[-1]})"

        query_x, T, length_padding = pad_seq(query_x, self.config)
        key_value_x, Tkv, _ = pad_seq(key_value_x, self.config)
        assert T % self.config.n_groups == 0, "query seq length must be a multiple of nb of groups"
        assert Tkv % self.config.n_groups == 0, "key seq length must be a multiple of nb of groups"
        lg = T // self.config.n_groups
        lgkv = Tkv // self.config.n_groups
        query_x = query_x.view(B, self.config.n_groups, lg, C)
        key_value_x = key_value_x.view(B, self.config.n_groups, lgkv, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.to_q(query_x)
        k, v = self.to_kv(key_value_x).split(self.n_embd, dim=-1)
        # q, k layer norm before attention
        q, k = self.ln_qk(q), self.ln_qk(k)
        q = q.view(B, self.config.n_groups, lg, self.n_head, self.n_embd // self.n_head).transpose(2, 3) # (B, ng, nh, lg, hs)
        k = k.view(B, self.config.n_groups, lgkv, self.n_head, self.n_embd // self.n_head).transpose(2, 3) # (B, ng, nh, lgkv, hs)
        v = v.view(B, self.config.n_groups, lgkv, self.n_head, self.n_embd // self.n_head).transpose(2, 3) # (B, ng, nh, lgkv, hs)

        if self.config.pos_embd == 'rope':
          q, k = self.apply_rope(q, k)

        # causal self-attention; Self-attend: (B, ng, nh, lg, hs) x (B, ng, nh, hs, lgkv) -> (B, ng, nh, lg, lgkv)
        # B, ng, nh, lg, lgkv
        attn_mask = causal_mask[:B, None, None, :lg, :lgkv] # T, Tkv can be padded
        causal_mask = None
        # large number of negatives values (bc of small number of key_indices behind query at the beginning) provokes instabilities
        # in the iterative softmax algo used by fused attention kernels in flash attention
        # see https://github.com/pytorch/pytorch/issues/110213
        y = compute_attn(q, k, v, self.attn_dropout, self.flash and not self.config.use_separate_routers_for_q_and_kv, self.config, self.training, attn_mask)
        y = y.transpose(2, 3).contiguous().view(B, self.config.n_groups, lg, self.n_embd) # re-assemble all head outputs side by side # (B, ng, lg, C)

        # output projection
        y = self.resid_dropout(self.proj_to_embd(y)).view(B, T, C)
        if length_padding != 0:
              y = y[:, :-length_padding, :]
        return y

class HeavyCausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = int(self.config.coeff_heavy_branch_n_embd * self.config.n_embd)
        assert self.n_embd % config.n_head == 0, f'heavy branch n_embed({self.n_embd}) should be a multiple of n_head({self.config.n_head})'
        self.nb_non_shared_soft_experts = self.config.nb_soft_experts - self.config.nb_shared_experts

        # key, query, value projections for all heads, but in a batch
        self.to_q = nn.Linear(self.n_embd, self.nb_non_shared_soft_experts * self.n_embd, bias=self.config.bias)
        if self.config.use_same_kv_for_soft_experts:
          self.to_kv = nn.Linear(self.n_embd, 2 * self.n_embd, bias=self.config.bias)
        else:
          self.to_kv = nn.Linear(self.n_embd, 2 * (self.n_embd * self.nb_non_shared_soft_experts), bias=self.config.bias)

        # output projection
        self.proj_to_embd = nn.Linear(self.n_embd, self.n_embd, bias=self.config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_dropout = nn.Dropout(self.config.dropout)
        self.n_head = self.config.n_head
        self.dropout = self.config.dropout
        self.ln_qk = RMSNorm(self.n_embd) if self.config.normalization == 'rmsnorm' else LayerNorm(self.n_embd, bias=self.config.bias)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")


    def rotate_embeddings(self, x):
        x = x.view(*x.shape[:-1], -1, 2).flip(-1) # divide by 2 the head size
        x[...,0] *= -1
        return x.flatten(start_dim=-2)

    def apply_rope(self, q, k):
      B, ng, nh, lg, hs = q.size()
      lgkv = k.shape[-2]
      pos = 10000**((-2 * torch.arange(0, hs, 2, device=x.device) - 1)/ hs)
      token_seq = torch.arange(lg, dtype=pos.dtype, device=x.device).unsqueeze(1) @ pos.unsqueeze(0)
      kv_token_seq = torch.arange(lgkv, dtype=pos.dtype, device=x.device).unsqueeze(1) @ pos.unsqueeze(0)
      pos = None
      rotary_embds = torch.cat((token_seq, token_seq), dim=-1)
      token_seq = None
      kv_rotary_embds = torch.cat((kv_token_seq, kv_token_seq), dim=-1)
      kv_token_seq = None
      q = (q * rotary_embds.cos()) + \
        (self.rotate_embeddings(q) * rotary_embds.sin())
      rotary_embds = None
      k = (k * kv_rotary_embds.cos()) + \
        (self.rotate_embeddings(k) * kv_rotary_embds.sin())
      return q, k

    def compute_qkv(self, query_x, key_value_x):
        B, T, C = query_x.size()
        Tkv = key_value_x.shape[1]
        q = self.to_q(query_x).view(B, T, self.nb_non_shared_soft_experts, self.n_embd).transpose(1, 2) # B, nse, T, C
        if self.config.use_same_kv_for_soft_experts:
          k, v  = self.to_kv(key_value_x).split(self.n_embd, dim=-1) # B, Tkv, n_embd
          # q, k layer norm before attention
          q, k = self.ln_qk(q), self.ln_qk(k)

          q = q.view(B, self.nb_non_shared_soft_experts, T, self.n_head, self.n_embd // self.n_head).transpose(2, 3) # B, nse, nh, T, hs
          k = k.view(B, Tkv, self.n_head, self.n_embd // self.n_head).transpose(1, 2).unsqueeze(1) # (B, 1, nh, Tkv, hs)
          v = v.view(B, Tkv, self.n_head, self.n_embd // self.n_head).transpose(1, 2).unsqueeze(1) # (B, 1, nh, Tkv, hs)
        else:
          k, v = self.to_kv(key_value_x).split(self.nb_non_shared_soft_experts * self.n_embd, dim=-1) # B, Tkv, nse * C
          k = k.view(B, Tkv, self.nb_non_shared_soft_experts, self.n_embd).transpose(1, 2) # B, nse, Tkv, C
          v = v.view(B, Tkv, self.nb_non_shared_soft_experts, self.n_embd).transpose(1, 2) # B, nse, Tkv, C
          # q, k layer norm before attention
          q, k = self.ln_qk(q), self.ln_qk(k)

          q = q.view(B, self.nb_non_shared_soft_experts, T, self.n_head, self.n_embd // self.n_head).transpose(2, 3) # B, nse, nh, T, hs
          k = k.view(B, self.nb_non_shared_soft_experts, Tkv, self.n_head, self.n_embd // self.n_head).transpose(2, 3) # B, nse, nh, Tkv, hs
          v = v.view(B, self.nb_non_shared_soft_experts, Tkv, self.n_head, self.n_embd // self.n_head).transpose(2, 3) # B, nse, nh, Tkv, hs
        return q, k, v

    def forward(self, query_x, key_value_x, causal_mask):
        B, T, C = query_x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        Tkv = key_value_x.shape[1]
        assert (B==key_value_x.shape[0] and C==key_value_x.shape[-1]), f"query_value_x and key_value_x should have the same batch({B}!={key_value_x.shape[0]}) and n_embd({C}!={key_value_x.shape[-1]})"


        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        assert C % self.n_head == 0, f"n_embd ({C}) must be a multiple of n_head ({self.n_head})"
        q, k, v = self.compute_qkv(query_x, key_value_x)

        if self.config.pos_embd == 'rope':
          q, k = self.apply_rope(q, k)


        # (B, nse, nh, T, Tkv)
        attn_mask = causal_mask.view(B, 1, 1, T, Tkv) # between batch we can mask different position
        causal_mask = None
        # large number of negatives values (bc of small number of key_indices behind query at the beginning) provokes instabilities (renders NaN)
        # in the iterative softmax algo used by fused attention kernels in flash attention
        # see https://github.com/pytorch/pytorch/issues/110213
        y = compute_attn(q, k, v, self.attn_dropout, self.flash and not self.config.use_separate_routers_for_q_and_kv, self.config, self.training, attn_mask)
        y = y.transpose(2, 3).contiguous().view(B, self.nb_non_shared_soft_experts, T, C) # (B, nse, T, C)

        # output projection
        return self.resid_dropout(self.proj_to_embd(y)) # B, nse, T, C

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        # key, query, value projections for all heads, but in a batch
        self.to_qkv = nn.Linear(config.n_embd, 3 * self.config.n_embd, bias=self.config.bias)
        # output projection
        self.proj_to_embd = nn.Linear(self.config.n_embd, self.config.n_embd, bias=self.config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_dropout = nn.Dropout(self.config.dropout)
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        self.dropout = self.config.dropout
        self.ln_qk = RMSNorm(self.n_embd) if self.config.normalization == 'rmsnorm' else LayerNorm(self.n_embd, bias=self.config.bias)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(self.config.block_size, self.config.block_size))
                                        .view(1, 1, self.config.block_size, self.config.block_size))

    def rotate_embeddings(self, x):
        x = x.view(*x.shape[:-1], -1, 2).flip(-1)
        x[...,0] *= -1
        return x.flatten(start_dim=-2)

    def apply_rope(self, q, k):
      B, nh, T, hs = q.size()
      pos = 10000**((-2 * torch.arange(0, hs, 2, device=x.device) - 1)/ hs)
      token_seq = torch.arange(T, dtype=pos.dtype, device=x.device).unsqueeze(1) @ pos.unsqueeze(0)
      pos = None
      rotary_embds = torch.cat((token_seq, token_seq), dim=-1)
      q = (q * rotary_embds.cos()) + \
        (self.rotate_embeddings(q) * rotary_embds.sin())
      k = (k * rotary_embds.cos()) + \
        (self.rotate_embeddings(k) * rotary_embds.sin())
      return q, k

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.to_qkv(x).split(self.n_embd, dim=2)
        # q, k layer norm before attention
        q, k = self.ln_qk(q), self.ln_qk(k)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if self.config.pos_embd == 'rope':
          q, k = self.apply_rope(q, k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attn_mask = None if self.flash else self.bias[:,:,:T,:T]
        y = compute_attn(q, k, v, self.attn_dropout, self.flash, self.config, self.training, attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        return self.resid_dropout(self.proj_to_embd(y))

class MLP(nn.Module):

    def __init__(self, config, block_type='vanilla'):
        super().__init__()
        self.config = config
        self.block_type = block_type
        assert block_type in {'vanilla', 'light', 'heavy'}, "only 'vanilla', 'light' and 'heavy' block types in MoD blocks"

        if self.block_type == 'vanilla':
          self.n_embd = self.config.n_embd
        elif self.block_type == 'light':
          self.n_embd = int(self.config.coeff_light_branch_n_embd * self.config.n_embd)
        else:
          self.n_embd = int(self.config.coeff_heavy_branch_n_embd * self.config.n_embd)

        if self.config.activation in {'swiglu', 'geglu', 'reglu'}:
          self.mlp_inner_proj = nn.Linear(self.n_embd, 2 * (4 * self.n_embd), bias=self.config.bias) # double hdim because gating
          self.activation = GatedActivation(activation=self.config.activation)
        else:
          self.mlp_inner_proj = nn.Linear(self.n_embd, 4 * self.n_embd, bias=self.config.bias)
          self.activation = nn.GELU()

        if (self.block_type == 'vanilla') or (self.block_type == 'light'):
          self.mlp_out_proj  = nn.Linear(4 * self.n_embd, self.n_embd, bias=config.bias)
        else:
          self.nb_non_shared_soft_experts = self.config.nb_soft_experts - self.config.nb_shared_experts
          assert self.nb_non_shared_soft_experts != 0, "should have at least one non shared expert in the heavy branch soft experts"
          self.soft_experts_out_dim = int(self.config.coeff_soft_expert_mlp_dim * self.n_embd // self.config.nb_soft_experts)
          self.non_shared_soft_experts = nn.ParameterList([nn.Parameter(torch.randn(4 * self.n_embd, self.soft_experts_out_dim),
                                                      requires_grad=True) for _ in range(self.nb_non_shared_soft_experts)])
          if config.nb_shared_experts != 0:
            self.shared_soft_experts = nn.ParameterList([nn.Parameter(torch.randn(4 * self.n_embd, self.soft_experts_out_dim),
                                                    requires_grad=True) for _ in range(config.nb_shared_experts)])
          self.soft_experts_proj = nn.Linear(self.soft_experts_out_dim, self.n_embd, self.config.bias)

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        x = self.activation(self.mlp_inner_proj(x))
        if (self.block_type == 'vanilla') or (self.block_type == 'light'):
          return self.dropout(self.mlp_out_proj(x))

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

        y = self.dropout(self.soft_experts_proj(y)) # B, nse, T, C
        return (y.softmax(dim=1) * y).sum(dim=1) # B, T, C


class VanillaBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = RMSNorm(config.n_embd) if config.normalization == 'rmsnorm' else LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd) if config.normalization == 'rmsnorm' else LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        if self.config.relufication:
          self.activation = nn.ReLU()

    def forward(self, x):
        x = x + self.attn(self.activation(self.ln_1(x))) if self.config.relufication else x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

def apply_top_k_selection(router_weights, k):
    """Applies top-k selection and sorting to router weights.

    Args:
        router_weights (torch.Tensor): Router weights to process.
        k (int): Number of top elements to select.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Top-k weights and indices.
    """
    top_k_weights, top_k_indices = torch.topk(router_weights, k, dim=1, sorted=False) # B, k, 1
    # top_k_weights = F.softmax(top_k_weights, dim=1) # normalize
    top_k_weights = F.sigmoid(top_k_weights) # parallel # B, k, 1
    sorted_indices = torch.argsort(top_k_indices, dim=1)
    top_k_indices = torch.gather(top_k_indices, dim=1, index=sorted_indices)
    top_k_weights = torch.gather(top_k_weights, dim=1, index=sorted_indices)
    return top_k_weights, top_k_indices

def calculate_sampling_predictor_loss(predictions, top_k_indices):
    """
    Calculates the sampling predictor loss for either query or key-value pairs.

    Args:
        predictions (torch.Tensor): Predictions from the sampling predictor.
        top_k_indices (torch.Tensor): Indices of the top-k elements from which sampling is performed.

    Returns:
        torch.Tensor: Sampling predictor loss or None if not applicable.
    """
    target = torch.zeros(*predictions.shape[:-1], device=predictions.device, dtype=predictions.dtype) # B, T
    target.scatter_(dim=1, index=top_k_indices.squeeze(-1), src=torch.ones_like(top_k_indices.squeeze(-1), device=predictions.device, dtype=predictions.dtype))
    loss = F.binary_cross_entropy_with_logits(predictions.squeeze(-1), target)
    return loss

class MoDBlock(nn.Module):
  def __init__(self, config, block_type='light'):
    super().__init__()
    assert block_type in {'light', 'heavy'}, "only 'light' and 'heavy' block types in MoD blocks"
    assert 0 <= config.light_branch_query_capacity < 1, "light branch query capacity should in [0, 1["
    assert 0 <= config.heavy_branch_query_capacity < 1, "heavy branch query capacity should in [0, 1["
    assert 0 <= config.light_branch_key_value_capacity < 1, "light branch key-value capacity should in [0, 1["
    assert 0 <= config.heavy_branch_key_value_capacity < 1, "heavy branch key-value capacity should in [0, 1["


    self.config = config
    self.block_type = block_type

    if self.block_type == 'light':
      self.n_embd = int(self.config.coeff_light_branch_n_embd * self.config.n_embd)
    else:
      self.n_embd = int(self.config.coeff_heavy_branch_n_embd * self.config.n_embd)

    self.query_router = nn.Linear(self.n_embd, 1, self.config.bias)
    self.query_sampling_predictor = nn.Linear(self.n_embd, 1, self.config.bias) # used at inf for causal routing. can free query_router at this moment
    if self.config.use_separate_routers_for_q_and_kv:
      self.key_value_router = nn.Linear(self.n_embd, 1, self.config.bias)
      self.key_value_sampling_predictor = nn.Linear(self.n_embd, 1, self.config.bias)
    self.already_free_router_weights = False # allows to track state in case we save the model

    self.in_proj = nn.Linear(self.config.n_embd, self.n_embd, self.config.bias)
    self.out_proj = nn.Linear(self.n_embd, self.config.n_embd, self.config.bias)
    self.out_proj.weight = nn.Parameter(self.in_proj.weight.t()) # tie

    self.ln_1 = RMSNorm(self.n_embd) if config.normalization == 'rmsnorm' else LayerNorm(self.n_embd, bias=self.config.bias)
    self.ln_2 = RMSNorm(self.n_embd) if self.config.normalization == 'rmsnorm' else LayerNorm(self.n_embd, bias=self.config.bias)

    if self.block_type == 'light':
      self.attn = LightCausalSelfAttention(self.config)
      self.query_capacity = self.config.light_branch_query_capacity
      self.query_top_k = int(self.query_capacity * self.config.block_size)
      if self.config.use_separate_routers_for_q_and_kv:
        self.key_value_capacity = self.config.light_branch_key_value_capacity
        self.key_value_top_k = int(self.key_value_capacity * self.config.block_size)
    else:
      self.attn = HeavyCausalSelfAttention(self.config)
      self.nb_non_shared_soft_experts = config.nb_soft_experts - config.nb_shared_experts
      self.query_capacity = self.config.heavy_branch_query_capacity
      self.query_top_k = int(self.query_capacity * self.config.block_size)
      if self.config.use_separate_routers_for_q_and_kv:
        self.key_value_capacity = self.config.heavy_branch_key_value_capacity
        self.key_value_top_k = int(self.key_value_capacity * self.config.block_size)
    self.mlp = MLP(self.config, block_type)
    if self.config.relufication:
          self.activation = nn.ReLU()

  @property
  def get_block_type(self):
    return self.block_type

  def free_router_weights(self):
    """
    Enables to free the router weights during inference time (do not use while cross-validating during training)
    """
    self.query_router = None
    if self.config.use_separate_routers_for_q_and_kv:
      self.key_value_router = None
    self.already_free_router_weights = True

  def forward(self, x): 
    assert not(self.training and self.already_free_router_weights), "should not free router weights during training"

    B, T, C = x.shape
    x = self.in_proj(x)

    # router weights
    query_router_weights = self.query_router(x) # B, T, 1
    with torch.no_grad():
      sampling_predictor_input = x.detach() # stop grad, use representation without interfering with main model learning
    query_sampling_predictions = self.query_sampling_predictor(sampling_predictor_input) 
    key_value_sampling_predictions = None
    if self.config.use_separate_routers_for_q_and_kv:
      key_value_router_weights = self.key_value_router(x)
      key_value_sampling_predictions = self.key_value_sampling_predictor(sampling_predictor_input) # B, T, 1
    sampling_predictor_input=None

    # top-k selection
    query_top_k_weights, query_top_k_indices = apply_top_k_selection(query_router_weights, min(self.query_top_k, int(self.query_capacity * T))) # B, k , 1
    causal_mask = query_top_k_indices >= query_top_k_indices.squeeze(-1)[:, None, :] # B, k, 1 vs B, 1, k # do it here just bc of the case different router for key
    if self.config.use_separate_routers_for_q_and_kv:
      key_value_top_k_weights, key_value_top_k_indices = apply_top_k_selection(key_value_router_weights, min(self.key_value_top_k, int(self.key_value_capacity * T)))
      # only attend to key behind us or at the same pos in the sequence
      causal_mask = query_top_k_indices >= key_value_top_k_indices.squeeze(-1)[:, None, :] # B, k, 1 vs B, 1, k (can be different k)

    # sampling predictor loss
    query_sampling_predictor_loss = calculate_sampling_predictor_loss(query_sampling_predictions, query_top_k_indices)
    query_sampling_predictions = None # free up memory
    key_value_sampling_predictor_loss = torch.tensor(0.0).to(x.device)
    if self.config.use_separate_routers_for_q_and_kv:
      key_value_sampling_predictor_loss = calculate_sampling_predictor_loss(key_value_sampling_predictions, key_value_top_k_indices)
      key_value_sampling_predictions = None


    # processing
    query_selected_tokens = torch.gather(x, dim=1, index=query_top_k_indices.expand(-1, -1, self.n_embd)) # B, k, n_embd
    if self.config.use_separate_routers_for_q_and_kv:
      key_value_selected_tokens = torch.gather(x, dim=1, index=key_value_top_k_indices.expand(-1, -1, self.n_embd))

    if self.config.use_separate_routers_for_q_and_kv:
      attended_tokens = self.attn(
          query_x = self.activation(self.ln_1(query_selected_tokens)) if self.config.relufication else self.ln_1(query_selected_tokens),
          # put router on the gradient path
          key_value_x = self.activation(self.ln_1(key_value_top_k_weights * key_value_selected_tokens)) if self.config.relufication else self.ln_1(key_value_top_k_weights * key_value_selected_tokens),
          causal_mask = causal_mask
      )
    else:
      attended_tokens = self.attn(
          query_x = self.activation(self.ln_1(query_selected_tokens)) if self.config.relufication else self.ln_1(query_selected_tokens),
          key_value_x = self.activation(self.ln_1(query_selected_tokens)) if self.config.relufication else self.ln_1(query_selected_tokens),
          causal_mask = causal_mask
      )

    causal_mask = None # free

    if self.block_type == 'light':
      processed_tokens = query_selected_tokens + self.mlp(self.ln_2(query_selected_tokens + attended_tokens)) # B, k, n_embd
    else:
      processed_tokens = query_selected_tokens + self.mlp(self.ln_2(query_selected_tokens.unsqueeze(1).expand(-1, self.nb_non_shared_soft_experts, -1, -1) + attended_tokens))

    if self.config.use_light_heavy_branch_at_same_layer_level:
      return self.out_proj(query_top_k_weights * processed_tokens), query_top_k_indices.expand(-1, -1, C), query_sampling_predictor_loss, key_value_sampling_predictor_loss

    # combine
    output = x.clone()
    output.scatter_add_( # in-place, res con for token that do not bypass the block computation
        dim=1,
        index=query_top_k_indices.expand(-1, -1, self.n_embd),
        src=query_top_k_weights * processed_tokens, # put router weights on the gradient path + broadcast values
    )

    return self.out_proj(output), query_sampling_predictor_loss, key_value_sampling_predictor_loss



@dataclass
class ConditionalTFConfig:
    block_size: int = 768
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_series_blocks: int = 6 # series : serie = successives light - heavy. at the end closed by vanilla tf (need to try also light-heavy same lvl)
    n_light_heavy_blocks_in_each_serie: int = 3 # x times (light - heavy) before vanilla tf
    n_head: int = 32
    n_embd: int = 256
    light_branch_query_capacity: float = 0.12 # percentage
    heavy_branch_query_capacity: float = 0.06
    use_separate_routers_for_q_and_kv: bool = False # separate tokens that requires info from those that posses such info
    light_branch_key_value_capacity: float = 0.12 # percentage
    heavy_branch_key_value_capacity: float = 0.06
    coeff_light_branch_n_embd: float = 1./2.
    coeff_heavy_branch_n_embd: float = 2.0
    use_same_kv_for_soft_experts: bool = False
    use_light_heavy_branch_at_same_layer_level: bool = True # if false they are used one after the other
    nb_soft_experts: int = 6
    nb_shared_experts: int = 2
    coeff_soft_expert_mlp_dim: int = 1 # each soft expert will have coeff_soft_expert_mlp_dim * n_embd / nb_soft_experts as output dim
    n_groups: int = 8 # size of the local attn in light csa
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    activation: str = 'swiglu' # gelu, swiglu, geglu, reglu?
    relufication: bool = False # https://arxiv.org/pdf/2310.04564 # only add relu after ln before attn. let choice for the mlps act
    normalization: str = 'rmsnorm' # layernorm or rmsnorm
    noisy_embd_alpha: float = 10.0 # 0 if you don't want to use noisy embedding # https://openreview.net/pdf?id=0bMmZ3fkCk
    nb_mem_token: int = 4 # nb memory / sink token # https://arxiv.org/pdf/2309.17453
    muP: bool = True # hyper-param transfer muP # https://arxiv.org/pdf/2203.03466
    pos_embd: str = 'simple' # simple, rope, no
    init_weight: str = 'nanogpt' # 'nanogpt', 'spike_no_more' # overrides if muP
    apply_gradient_checkpointing: bool = False # trade compute for memory # not implemented yet
    gradient_checkpoint_sequential_factor: int = 2

def init_with_muP(pn, p, config):
  special_weights = ['lm_head.weight', 'shared_soft_experts', 'wpe', 'wte']
  total_n_layer = 1 + config.n_series_blocks * (2 * config.n_light_heavy_blocks_in_each_serie) +  config.n_series_blocks * 1
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
  total_n_layer = 1 + config.n_series_blocks * (2 * config.n_light_heavy_blocks_in_each_serie) +  config.n_series_blocks * 1
  coeff_std = math.sqrt(2 / 5) / math.sqrt(2 * total_n_layer)
  if any(subname in pn for subname in ['proj_to_embd', 'out_proj']):
    fan_in = p.shape[1] # pytorch default linear shape: (out_features, in_features)
    torch.nn.init.normal_(p, mean=0.0, std=coeff_std/(math.sqrt(fan_in)))
  elif 'shared_soft_experts' in pn: # shared and non_shared
    fan_in = p.shape[0] # (in_features, out_features)
    torch.nn.init.normal_(p, mean=0.0, std=coeff_std/(math.sqrt(fan_in)))

def init_with_nano_gpt(pn, p, config):
  total_n_layer = 1 + config.n_series_blocks * (2 * config.n_light_heavy_blocks_in_each_serie) +  config.n_series_blocks * 1
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

class ConditionalTF(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        h = nn.ModuleList([VanillaBlock(self.config)])
        for i in range(self.config.n_series_blocks):
          for j in range(self.config.n_light_heavy_blocks_in_each_serie):
            if self.config.use_light_heavy_branch_at_same_layer_level: 
              h.append(nn.ModuleList([MoDBlock(self.config, 'light'), MoDBlock(self.config, 'heavy')]))
            else: 
              h.append(MoDBlock(self.config, 'light'))
              h.append(MoDBlock(self.config, 'heavy'))
          h.append(VanillaBlock(self.config))

        self.blocks = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            drop = nn.Dropout(self.config.dropout),
            h = h,
            ln_f = RMSNorm(self.config.n_embd) if config.normalization == 'rmsnorm' else LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))
        self.lm_head = nn.Linear(self.config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.blocks.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        if self.config.nb_mem_token != 0:
          self.mem_token = nn.Parameter(torch.randn(self.config.nb_mem_token, self.config.n_embd), # [batch_size, nb_memory_token, n_embd]
                           requires_grad=True) # make sure the embedding is learnable

        self.n_layer = 1 + self.config.n_series_blocks * (2 * self.config.n_light_heavy_blocks_in_each_serie) +  self.config.n_series_blocks * 1


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
        n_params = sum(p.numel() for pn, p in self.named_parameters() if 'sampling_predictor' not in pn)
        if non_embedding:
            n_params -= (self.blocks.wpe.weight.numel() + self.config.nb_mem_token) if self.config.nb_mem_token != 0 else self.blocks.wpe.weight.numel()

        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
          if self.config.muP: # must deal with it first
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(self.config.n_embd)) # muP
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
            torch.nn.init.normal_(module.weight, mean=0.0, std= math.sqrt(2 / (5 * self.config.n_embd))) # account for tok embd and all pos embd
          else: # nanoGPT as default
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        if self.config.init_weight == 'spike_no_more':
          tok_emb = self.blocks.wte(idx) * self.config.n_embd # token embeddings of shape (b, t, n_embd) # scaled embed
        else:
          tok_emb = self.blocks.wte(idx)

        if (self.config.noisy_embd_alpha != 0.0):
          eps = 2 * torch.rand(1) - 1
          tok_emb += ((self.config.noisy_embd_alpha / math.sqrt(tok_emb.size(-2) * tok_emb.size(-1))) * eps).to(device)

        if self.config.pos_embd == 'no' or self.config.pos_embd == 'rope':
          pos_emb = torch.zeros(t, self.config.n_embd, dtype=torch.long, device=device) # (t, n_embd)
        else:
          pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
          pos_emb = self.blocks.wpe(pos) # position embeddings of shape (t, n_embd)

        # add memory/sink token after the embedding if requires
        if self.config.nb_mem_token != 0:
          mem_token = self.mem_token.expand(b, -1, -1).to(device)
        x = self.blocks.drop(torch.cat((mem_token, tok_emb + pos_emb), dim=1)) if self.config.nb_mem_token != 0 else self.blocks.drop(tok_emb + pos_emb)
        pos_emb, tok_emb = None, None

        losses = {'CE': torch.tensor(0.0).to(device)}

        for i, block in enumerate(self.blocks.h):
          if isinstance(block, nn.ModuleList):
              # first treat light branch
              light_branch_procesed_tokens, light_branch_index_selected_tokens, light_branch_query_sampling_predictor_loss, light_branch_key_value_sampling_predictor_loss = block[0](x)
              losses[f'query_sampling_predictor_loss_block_{i}_type_{block[0].get_block_type}'] = light_branch_query_sampling_predictor_loss
              losses[f'key_value_sampling_predictor_loss_block_{i}_type_{block[0].get_block_type}'] = light_branch_key_value_sampling_predictor_loss
              light_branch_query_sampling_predictor_loss, light_branch_key_value_sampling_predictor_loss = None, None

              # manage heavy branch  
              heavy_branch_procesed_tokens, heavy_branch_index_selected_tokens, heavy_branch_query_sampling_predictor_loss, heavy_branch_key_value_sampling_predictor_loss = block[1](x)
              losses[f'query_sampling_predictor_loss_block_{i}_type_{block[1].get_block_type}'] = heavy_branch_query_sampling_predictor_loss
              losses[f'key_value_sampling_predictor_loss_block_{i}_type_{block[1].get_block_type}'] = heavy_branch_key_value_sampling_predictor_loss
              heavy_branch_query_sampling_predictor_loss, heavy_branch_key_value_sampling_predictor_loss = None, None

              # add representations tokens processed by the light branch
              x.scatter_add_( # in-place, res con for token that do not bypass the block computation
                  dim=1,
                  index=light_branch_index_selected_tokens,
                  src=light_branch_procesed_tokens.to(x.dtype), 
              )
              light_branch_procesed_tokens, light_branch_index_selected_tokens = None, None

              # add representations tokens processed by the heavy branch
              x.scatter_add_( 
                  dim=1,
                  index=heavy_branch_index_selected_tokens,
                  src=heavy_branch_procesed_tokens.to(x.dtype), 
              )
              heavy_branch_procesed_tokens, heavy_branch_index_selected_tokens = None, None

          elif isinstance(block, MoDBlock): # (need to try also light-heavy same lvl)
            x, query_sampling_predictor_loss, key_value_sampling_predictor_loss = block(x)
            losses[f'query_sampling_predictor_loss_block_{i}_type_{block.get_block_type}'] = query_sampling_predictor_loss
            losses[f'key_value_sampling_predictor_loss_block_{i}_type_{block.get_block_type}'] = key_value_sampling_predictor_loss
            query_sampling_predictor_loss, key_value_sampling_predictor_loss = None, None

          else:
            x = block(x)

        x = self.blocks.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) / self.config.n_embd if self.config.muP else self.lm_head(x) # muP
            # discard memory token if any
            if self.config.nb_mem_token != 0:
              logits = logits[:,self.config.nb_mem_token:,:]
              loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
              loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            losses['CE'] = loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim # only want to predict the char for the last elm of the seq, use [-1] to keep the seq dim when slicing
            logits = logits / self.config.n_embd if self.config.muP else logits # muP # doesn't impact in fact
            losses = None

        return logits, losses

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.blocks.wpe.weight = nn.Parameter(self.blocks.wpe.weight[:block_size])
        for block in self.blocks.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, use_8_bit_optim, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if 'sampling_predictor' not in pn}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        # use seperate optimizer bc probably different learning dynamic
        sampling_predictor_params = {pn: p for pn, p in self.named_parameters() if 'sampling_predictor' in pn}
        decay_sampling_predictor_params = [p for _, p in sampling_predictor_params.items() if p.dim() >= 2]
        no_decay_sampling_predictor_params = [p for _, p in sampling_predictor_params.items() if p.dim() < 2]

        print(f"using muP: {self.config.muP}")


        if self.config.muP:
          # take care of regular param £
          special_weights = ['wpe', 'wte', 'shared_soft_experts']
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

          # take care of predictor params
          sampling_predictor_decay_fan_in_params = {p.shape[1]: [] for n, p in sampling_predictor_params.items() if p.dim() >= 2}
          sampling_predictor_decay_fan_in_params_names = []
          for pn, p in sampling_predictor_params.items():
            if p.dim() >= 2:
              sampling_predictor_decay_fan_in_params[p.shape[1]].append(p)
              sampling_predictor_decay_fan_in_params_names.append(pn)


          linear_decay_params_group = [
              {'params': linear_decay_fan_in_params[fan_in], 'weight_decay': weight_decay[0], 'muPScale': fan_in, 'lr': learning_rate[0] / fan_in} for fan_in in linear_decay_fan_in_params.keys()
              ]

          soft_expert_linear_decay_params_groups = [
              {'params': soft_expert_linear_decay_fan_in_params[fan_in], 'weight_decay': weight_decay[0], 'muPScale': fan_in, 'lr': learning_rate[0] / fan_in} for fan_in in soft_expert_linear_decay_fan_in_params.keys()
              ]

          others_params_groups =  [
              {'params': others_params_decay, 'weight_decay': weight_decay[0], 'muPScale': 1.0},
              {'params': nodecay_params, 'weight_decay': 0.0, 'muPScale': 1.0}
          ]

          sampling_predictor_params_group = [
              {'params': sampling_predictor_decay_fan_in_params[fan_in], 'weight_decay': weight_decay[1], 'muPScale': fan_in, 'lr': learning_rate[1] / fan_in} for fan_in in sampling_predictor_decay_fan_in_params.keys()
          ]

          optim_groups = linear_decay_params_group + soft_expert_linear_decay_params_groups + others_params_groups
          optim_groups_sampling_predictor = sampling_predictor_params_group + [
              {'params': no_decay_sampling_predictor_params, 'weight_decay': 0.0, 'muPScale': 1.0}
          ]


          num_decay_params = (
              sum(p.numel() for fan_in in linear_decay_fan_in_params.keys() for p in linear_decay_fan_in_params[fan_in]) 
              + 
              sum(p.numel() for fan_in in soft_expert_linear_decay_fan_in_params.keys() for p in soft_expert_linear_decay_fan_in_params[fan_in]) 
              + 
              sum(p.numel() for p in others_params_decay)
              +
              sum(p.numel() for fan_in in sampling_predictor_decay_fan_in_params.keys() for p in sampling_predictor_decay_fan_in_params[fan_in])
          )
          num_nodecay_params = sum(p.numel() for p in nodecay_params) + sum(p.numel() for p in no_decay_sampling_predictor_params)
          num_tensors_decay_params = (
              sum(len(linear_decay_fan_in_params[fan_in]) for fan_in in linear_decay_fan_in_params.keys()) 
              + 
              sum(len(soft_expert_linear_decay_fan_in_params[fan_in]) for fan_in in soft_expert_linear_decay_fan_in_params.keys()) 
              + 
              len(others_params_decay)
              +
              sum(len(sampling_predictor_decay_fan_in_params[fan_in]) for fan_in in sampling_predictor_decay_fan_in_params.keys()) 
          )
          print("names linear params with decay: ", linear_decay_params_names)
          print("names soft experts mlp linear params with decay: ", soft_expert_linear_decay_params_names)
          print("names others params with decay: ", others_params_names_decay)
          print("names sampling predictor params with decay: ", sampling_predictor_decay_fan_in_params_names)
          print("test len decay param: ", num_tensors_decay_params == (len(decay_params)+len(decay_sampling_predictor_params)))

          print(f"num decayed parameter tensors: {num_tensors_decay_params}, with {num_decay_params:,} parameters")
          print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        else:
          optim_groups = [
              {'params': decay_params, 'weight_decay': weight_decay[0]},
              {'params': nodecay_params, 'weight_decay': 0.0}
          ]
          optim_groups_sampling_predictor = [
              {'params': decay_sampling_predictor_params, 'weight_decay': weight_decay[1]},
              {'params': no_decay_sampling_predictor_params, 'weight_decay': 0.0}
          ]
          num_decay_params = sum(p.numel() for p in decay_params) + sum(p.numel() for p in decay_sampling_predictor_params)
          num_nodecay_params = sum(p.numel() for p in nodecay_params) + sum(p.numel() for p in no_decay_sampling_predictor_params)
          print(f"num decayed parameter tensors: {len(decay_params)+len(decay_sampling_predictor_params)}, with {num_decay_params:,} parameters")
          print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # Fused AdamW is a faster and more efficient implementation of the AdamW optimizer in PyTorch
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict() # use the fused version if it is available
        if not(use_8_bit_optim[0]):
          optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate[0], betas=betas[0], **extra_args)
          print(f"using fused AdamW for regular params: {use_fused}")
        else:
          optimizer = AdamW8bit(optim_groups, lr=learning_rate[0], betas=betas[0], weight_decay=0.0) # wd set on param group, don't override with their arg

        if not(use_8_bit_optim[1]):
          optimizer_sampling_predictor = torch.optim.AdamW(optim_groups_sampling_predictor, lr=learning_rate[1], betas=betas[1], **extra_args)
          print(f"using fused AdamW for sampling predictor params: {use_fused}")  
        else:
          optimizer_sampling_predictor = AdamW8bit(optim_groups_sampling_predictor, lr=learning_rate[1], betas=betas[1], weight_decay=0.0)

        return optimizer, optimizer_sampling_predictor

    # def estimate_mfu(self, fwdbwd_per_iter, dt):
    #     """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    #     # first estimate the number of flops we do per iteration.
    #     # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    #     N = self.get_num_params()
    #     cfg = self.config
    #     L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
    #     flops_per_token = 6*N + 12*L*H*Q*T
    #     flops_per_fwdbwd = flops_per_token * T
    #     flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    #     # express our flops throughput as ratio of A100 bfloat16 peak flops
    #     flops_achieved = flops_per_iter * (1.0/dt) # per second
    #     flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
    #     mfu = flops_achieved / flops_promised
    #     return mfu

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
