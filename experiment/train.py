import torch
import torch.nn as nn
import torch.optim
import numpy as np
from scipy.io import loadmat
import perf_rnn
import transformer
import os
import glob
import util
import time
from datetime import datetime

ROOTP = os.getcwd()
DEBUG = True
IS_CLUSTER = True
LOAD_AGGREGATE = True
IS_RNN = False

def save_model(run_id, model, hp, best=False):
  best_str = 'best' if best else ''
  save_p = os.path.join(ROOTP, 'data/models/{}-{}-{}-checkpoint.pth'.format(run_id, best_str, model.get_name()))
  torch.save({
    'model_state_dict': model.state_dict(),
    'hp': hp
  }, save_p)

def get_source_files():
  fs = glob.glob(os.path.join(ROOTP, 'data/maestro-performance-events/*.mat'))
  if True:
    fs.append(os.path.join(ROOTP, 'data/my-midi/eg.mat'))
  return fs

def load_indices():
  return torch.load(os.path.join(ROOTP, 'data/maestro/maestro-aggregate.pth'))['indices']

def gen_batch(inds, batch_size, seq_len, device):
  bi = torch.zeros(batch_size, seq_len).type(torch.int64).to(device)
  ti = torch.zeros(batch_size, seq_len).type(torch.int64).to(device)
  for i in range(batch_size):
    seq = int(np.floor(np.random.rand() * len(inds)))
    i0 = int(np.floor((len(inds[seq]) - seq_len - 1) * np.random.rand()))
    i1 = i0 + seq_len
    bi[i, :] = inds[seq][i0:i1]
    ti[i, :] = inds[seq][(i0+1):(i1+1)]
  return bi, ti

def load_dataset():
  if LOAD_AGGREGATE:
    inds = load_indices()
    if DEBUG:
      inds = inds[:10]

  else:
    files = get_source_files()
    if DEBUG:
      files = files[:10]

    inds = []
    for i, file in enumerate(files):
      print('{} of {}'.format(i+1, len(files)), flush=True)
      f = loadmat(file)
      inds.append(torch.tensor(util.parse_performance_event_indices(f)))

  ntrain = int(np.floor(len(inds) * 0.7))
  nvalid = max(1, int(np.floor(len(inds) * 0.15)))
  train_inds = inds[:ntrain]
  valid_inds = inds[ntrain:ntrain+nvalid]
  test_inds = inds[ntrain+nvalid:]
  return train_inds, valid_inds, test_inds

def set_learn_rate(optim: torch.optim.Optimizer, lr: float) -> None:
  for param_group in optim.param_groups:
    param_group['lr'] = lr

def train_step(model, is_rnn, optim, crit, bi, ti):
  if is_rnn:
    res, _ = model(bi)
  else:
    res = model(bi)

  pr = res.flatten(0, 1)
  ti = ti.flatten(0, 1)
  
  optim.zero_grad()
  loss = crit(pr, ti)
  loss.backward()
  # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
  optim.step()
  acc = torch.sum(torch.argmax(pr, 1) == ti) / len(ti)
  return loss.item(), acc.item()

def evaluate(model, is_rnn, crit, bi, ti):
  if is_rnn:
    res, _ = model(bi)
  else:
    res = model(bi)
  pr = res.flatten(0, 1)
  ti = ti.flatten(0, 1)
  valid_loss = crit(pr, ti)
  acc = torch.sum(torch.argmax(pr, 1) == ti) / len(ti)
  return valid_loss.item(), acc.item()

def main():
  T0 = time.time()
  RUN_ID = datetime.now().strftime('%m%d%y_%H_%M_%S')

  device = util.get_device(IS_CLUSTER)
  print('Using device: ', device, flush=True)

  conv_conf = util.MidiConversionConfig()
  train_inds, valid_inds, test_inds = load_dataset()

  is_rnn = IS_RNN
  if is_rnn:
    hp = perf_rnn.hparams()
    model = perf_rnn.create_model_from_hparams(device, conv_conf.range, hp)

    batch_size = 64
    num_epochs = 32
    batches_per_epoch = 128
    lr = 0.002

  else:
    hp = transformer.hparams(use_rel_pos=False, is_cluster=IS_CLUSTER)
    model = transformer.create_model_from_hparams(device, conv_conf.range, hp)

    batch_size = 2
    lr = 1e-4
    num_epochs = 120
    batches_per_epoch = 256

  seq_len = hp['seq_len'] if 'seq_len' in hp else 1024

  if DEBUG:
    batches_per_epoch = 8
    num_epochs = 1

  train_inds = list(filter(lambda x: len(x) > seq_len, train_inds))
  test_inds = list(filter(lambda x: len(x) > seq_len, test_inds))
  valid_inds = list(filter(lambda x: len(x) > seq_len, valid_inds))

  model.train()
  crit = nn.CrossEntropyLoss()
  optim = torch.optim.Adam(model.parameters(), lr=lr)

  print('Model: ', model.get_name(), 'HParams: ', hp, flush=True)

  best_acc = 0.
  for ep in range(num_epochs):
    model.train()

    for i in range(batches_per_epoch):
      t0 = time.time()

      bi, ti = gen_batch(train_inds, batch_size, seq_len, device)
      loss, acc = train_step(model, is_rnn, optim, crit, bi, ti)

      tstep = time.time() - t0
      print('{} of {}; Loss: {}; Acc: {}; T: {}; T_Total: {}'.format(
        i+1, batches_per_epoch, loss, acc, tstep, time.time() - T0), flush=True)
    
    model.eval()
    if not DEBUG:
      save_model(RUN_ID, model, hp)

    with torch.no_grad():
      bi, ti = gen_batch(valid_inds, batch_size, seq_len, device)
      valid_loss, acc = evaluate(model, is_rnn, crit, bi, ti)
      print('{} of {} | Valid Loss: {}; Acc: {}'.format(ep+1, num_epochs, valid_loss, acc), flush=True)

      if not DEBUG and acc > best_acc:
        best_acc = acc
        save_model(RUN_ID, model, hp, best=True)

if __name__ == '__main__':
  main()