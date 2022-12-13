import torch
import os
import util

ROOTP = os.getcwd()
DSTP = os.path.join(ROOTP, 'data/jsb/midi')

if __name__ == '__main__':
  agg = torch.load(os.path.join(ROOTP, 'data/jsb/jsb-aggregate.pth'))['indices']
  for i, a in enumerate(agg):
    evts = util.MidiConversionConfig().parse_events(a)
    util.write_midi_file(evts, os.path.join(DSTP, '{}.mid'.format(i)))
