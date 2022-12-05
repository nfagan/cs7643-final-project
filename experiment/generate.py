import transformer
import perf_rnn
from util import MidiConversionConfig, get_device, parse_performance_event_indices, write_midi_file
import torch
import numpy as np
from scipy.io import loadmat
import os
import itertools

ROOTP = os.getcwd()

def load_model(device, p, is_rnn):
  load_res = torch.load(p)
  if 'hp' in load_res:
    hp = load_res['hp']
  else:
    hp = perf_rnn.hparams() if is_rnn else transformer.hparams(is_cluster=True)

  vocab_size = MidiConversionConfig().range

  if is_rnn:
    model = perf_rnn.create_model_from_hparams(device, vocab_size, hp)
  else:
    model = transformer.create_model_from_hparams(device, vocab_size, hp)

  model.load_state_dict(load_res['model_state_dict'])
  model.eval()
  return model

def load_prime_sequence(device, prime_p):
  prime_seq = torch.tensor(parse_performance_event_indices(loadmat(prime_p))).type(torch.int64)
  prime_seq = prime_seq.reshape(1, len(prime_seq)).to(device)
  return prime_seq

def sample_generate(model, is_rnn, prime_seq, seq_len, device, verbose=False):
  nprime = prime_seq.shape[1]
  assert seq_len >= nprime
  cell = None
  for i in range(seq_len - nprime):
    if verbose:
      print('{} of {}'.format(i + 1, seq_len - nprime))

    if is_rnn:
      prime_res, cell = model(prime_seq, cell)
    else:
      prime_res = model(prime_seq)
    ps = torch.softmax(prime_res, -1).squeeze()[-1, :].detach().cpu().numpy()

    next_seq = torch.zeros(1, nprime + i + 1).type(prime_seq.dtype).to(device)
    next_seq[0, :-1] = prime_seq
    next_seq[0, -1] = np.random.choice(np.arange(len(ps)), p=ps)
    prime_seq = next_seq
  return prime_seq.squeeze()

def main():
  prime_fnames = ['src3', 'eg2']
  model_fnames = [
    'rpr-transformer-checkpoint.pth',
    'transformer-checkpoint.pth'
    '120522_10_23_58-performance_rnn-checkpoint.pth',
    'jsb-rpr-transformer-checkpoint.pth',
    # todo: 'jsb-transformer-checkpoint.pth',
    'jsb-performance_rnn-checkpoint.pth',
  ]
  
  target_seq_len = 1024
  seed = 0

  for p in itertools.product(prime_fnames, model_fnames):
    prime_fname = p[0]
    model_fname = p[1]

    device = get_device()
    is_rnn = not ('transformer' in model_fname)

    model = load_model(device, os.path.join(ROOTP, 'data/models', model_fname), is_rnn)

    prime_p = os.path.join(ROOTP, 'data/my-midi/{}.mat'.format(prime_fname))
    prime_seq = load_prime_sequence(device, prime_p)

    if seed is not None:
      np.random.seed(seed)

    res = sample_generate(model, is_rnn, prime_seq, target_seq_len, device, verbose=True)
    res = MidiConversionConfig().parse_events(res.cpu().numpy())

    dst_p = os.path.join(ROOTP, 'data/my-midi/{}-{}-gen.mid'.format(model_fname, prime_fname))
    write_midi_file(res, dst_p)

if __name__ == '__main__':
  main()