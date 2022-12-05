import torch
import torch.nn as nn

class LSTMLayer(nn.Module):
  def __init__(self, in_dim, hs) -> None:
    super().__init__()
    self.lstm = nn.LSTM(in_dim, hs, 3, batch_first=True)

  def forward(self, x, inputs):
    y, h = self.lstm(x, inputs)
    return y, h

class PerformanceRNNModel(nn.Module):
  def __init__(self, vocab_size, embed_dim, hs) -> None:
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.lstm = LSTMLayer(embed_dim, hs)
    self.output = nn.Linear(hs, vocab_size)

  def forward(self, x, cell=None):
    y = self.embedding(x)
    y, cell = self.lstm(y, cell)
    out = self.output(y)
    return out, cell

  def get_name(self):
    return 'performance_rnn'

def generate_sequence(model, token0, seq_len, device):
  tok = torch.tensor(token0).reshape(1, 1).to(device)
  t0, cell = model(tok)
  res = []
  for i in range(seq_len):
    val = torch.argmax(t0.squeeze()).item()
    res.append(val)
    tok[0, 0] = val
    t0, cell = model(tok, cell)
  return torch.tensor(res)

def create_model_from_hparams(device, vocab_size, p):
  return PerformanceRNNModel(vocab_size, p['embed_dim'], p['hs']).to(device)

def hparams():
  embed_dim = 128
  hs = 512
  return {
    'embed_dim': embed_dim,
    'hs': hs,
  }