import torch
import torch.nn as nn
import numpy as np

def triu_mask(n):
  return torch.triu(torch.ones(n, n), diagonal=1)

def split_head(x, nh):
  assert x.shape[-1] % nh == 0
  return x.view(*x.shape[:2], nh, x.shape[-1] // nh).transpose(1, 2)

def srel_skew(x):
  v = nn.functional.pad(x, [1, 0])
  y = v.reshape(*v.shape[:2], v.shape[2]+1, v.shape[2])
  y = y[:, :, :x.shape[2], :]
  return y

class Attention(nn.Module):
  def __init__(self, dim, nh, rel_embed_dist, device) -> None:
    assert dim % nh == 0
    super().__init__()
    bias = False
    self.q = nn.Linear(dim, dim, bias=bias)
    self.k = nn.Linear(dim, dim, bias=bias)
    self.v = nn.Linear(dim, dim, bias=bias)
    self.proj = nn.Linear(dim, dim)
    self.nh = nh
    self.device = device
    self.triu_mask = None
    if rel_embed_dist > 0:
      self.rel_dist_embed = nn.Embedding(rel_embed_dist, dim)
    else:
      self.rel_dist_embed = None

  def get_rel_distance_embed(self, seq_len):
    if self.rel_dist_embed is None:
      return None
    else:
      max_len = self.rel_dist_embed.weight.shape[0]
      assert seq_len <= max_len
      ns = torch.arange(0, seq_len, device=self.device) + (max_len - seq_len)
      return split_head(self.rel_dist_embed(ns).unsqueeze(0), self.nh)

  def forward(self, x):
    n = x.shape[1]

    qx = split_head(self.q(x), self.nh)
    kx = split_head(self.k(x), self.nh)
    vx = split_head(self.v(x), self.nh)
    ex = self.get_rel_distance_embed(x.shape[1])

    if ex is None:
      srel = torch.zeros(1, 1, n, n, device=self.device)
    else:
      srel = srel_skew(qx @ ex.transpose(-1, -2))

    inv_dim = 1. / np.sqrt(x.shape[-1] / self.nh)
    qk = (qx @ kx.transpose(-1, -2) + srel) * inv_dim

    if self.triu_mask is None or self.triu_mask.shape[-1] != n:
      self.triu_mask = 1e9 * triu_mask(n).reshape(1, 1, n, n).to(self.device)
    atten = (torch.softmax(qk - self.triu_mask, -1) @ vx).transpose(1, 2).reshape(x.shape)
    return self.proj(atten)

class Transformer(nn.Module):
  def __init__(self, dh, rel_embed_dist, nh=4, hs_ff=512, device=None, dropout=0.3) -> None:
    super().__init__()

    self.atten_dropout = nn.Dropout(dropout)
    self.heads = Attention(dh, nh, rel_embed_dist, device)
    self.head_norm = nn.LayerNorm(dh)
    
    self.ff_lin1 = nn.Linear(dh, hs_ff)
    self.ff_lin2 = nn.Linear(hs_ff, dh)
    self.ff_norm = nn.LayerNorm(dh)
    self.ff_dropout = nn.Dropout(dropout)

  def attention(self, seq):
    atten = self.atten_dropout(self.heads(seq))
    return self.head_norm(seq + atten)

  def feedforward(self, atten):
    l1 = self.ff_lin1(atten)
    l1 = torch.relu(l1)
    l2 = self.ff_dropout(self.ff_lin2(l1))
    return self.ff_norm(l2 + atten)

  def forward(self, seq):
    atten = self.attention(seq)
    return self.feedforward(atten)

class TransformerModel(nn.Module):
  def __init__(self, device, num_decoder_layers, vocab_size, embed_dim, num_heads, hs_ff, seq_len, rel_embed_dist, dropout, disable_pos_embed=False) -> None:
    super().__init__()
    self.embed = nn.Embedding(vocab_size, embed_dim)
    self.pos_embed = None if disable_pos_embed else nn.Embedding(seq_len, embed_dim)
    self.transformer = nn.Sequential(
      *[Transformer(embed_dim, rel_embed_dist, hs_ff=hs_ff, nh=num_heads, device=device, dropout=dropout) for _ in range(num_decoder_layers)]
    )
    self.output = nn.Linear(embed_dim, vocab_size)
    self.seq_len = seq_len
    self.device = device

  def get_embedding(self, x):
    if self.pos_embed is not None:
      sample_range = torch.arange(0, x.shape[1]).repeat(x.shape[0], 1).to(self.device)
      pe = self.pos_embed(sample_range)
      return pe + self.embed(x)
    else:
      return self.embed(x)

  def forward(self, x):
    return self.output(self.transformer(self.get_embedding(x)))

  def get_name(self):
    return 'transformer'

def create_model_from_hparams(device, vocab_size, p):
  get = lambda d, k, v: v if k not in d else d[k]

  model = TransformerModel(
    device, p['num_decoder_layers'], vocab_size, 
    p['embed_dim'], p['num_heads'], p['hs_ff'], 
    p['seq_len'], p['rel_embed_dist'], p['dropout'], 
    disable_pos_embed=get(p, 'disable_pos_embed', False)).to(device)
  return model

def hparams(use_rel_pos=True, disable_pos_embed=False, is_cluster=False):
  seq_len = 1024
  embed_dim = 128
  num_heads = 4
  dropout = 0.1
  rel_embed_dist = seq_len if use_rel_pos else 0
  num_decoder_layers = 1
  hs_ff = 512

  if is_cluster:
    seq_len = 2048
    rel_embed_dist = seq_len if use_rel_pos else 0
    num_decoder_layers = 6
    num_heads = 8
    embed_dim = 512
    hs_ff = 1024

  return {
    'seq_len': seq_len,
    'embed_dim': embed_dim,
    'num_heads': num_heads,
    'num_decoder_layers': num_decoder_layers,
    'dropout': dropout,
    'rel_embed_dist': rel_embed_dist,
    'disable_pos_embed': disable_pos_embed,
    'hs_ff': hs_ff
  }