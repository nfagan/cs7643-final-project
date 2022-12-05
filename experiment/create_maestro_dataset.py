import torch
from scipy.io import loadmat
import os
import glob
import util

ROOTP = os.getcwd()

def save_indices(inds):
  save_p = os.path.join(ROOTP, 'data/maestro/maestro-aggregate.pth')
  torch.save({
    'indices': inds,
  }, save_p)

def get_source_files():
  fs = glob.glob(os.path.join(ROOTP, 'data/maestro-performance-events/*.mat'))
  if True:
    fs.append(os.path.join(ROOTP, 'data/my-midi/eg.mat'))
  return fs

if __name__ == '__main__':
  files = get_source_files()
  inds = []
  for i, file in enumerate(files):
    print('{} of {}'.format(i+1, len(files)))
    f = loadmat(file)
    inds.append(torch.tensor(util.parse_performance_event_indices(f)))
  save_indices(inds)