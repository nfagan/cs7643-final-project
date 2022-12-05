import music21
import numpy as np
import torch

class MidiConversionConfig(object):
  def __init__(self) -> None:
    num_notes = 128
    self.num_vel_bins = 32
    self.ts_interval = 1e-2
    self.num_time_bins = 100
    self.num_notes = num_notes
    self.ts_range =         [0, self.num_time_bins]
    self.note_on_range =    [self.ts_range[1], self.ts_range[1] + num_notes]
    self.note_off_range =   [self.note_on_range[1], self.note_on_range[1] + num_notes]
    self.vel_range =        [self.note_off_range[1], self.note_off_range[1] + self.num_vel_bins]
    self.range =            self.vel_range[1]

  def parse_event(self, i):
    assert i >= 0 and i < self.range and np.floor(i) == i
    if i >= self.ts_range[0] and i < self.ts_range[1]:
      return {'type': 'ts', 'value': (i - self.ts_range[0] + 1) * self.ts_interval}
    elif i >= self.note_on_range[0] and i < self.note_on_range[1]:
      return {'type': 'on', 'value': int(i - self.note_on_range[0])}
    elif i >= self.note_off_range[0] and i < self.note_off_range[1]:
      return {'type': 'off', 'value': int(i - self.note_off_range[0])}
    else:
      return {'type': 'vel', 'value': int((i - self.vel_range[0]) * (128 / self.num_vel_bins))}

  def parse_events(self, inds):
    res = []
    for i in range(len(inds)):
      res.append(self.parse_event(inds[i]))
    return res

def to_midi_events(mt, events):
  note_vel = 0
  ticks_per_second = 2050  # TODO
  dt = 0

  def make_note_event(evt, chan, note_vel):
    is_on = evt['type'] == 'on'
    assert evt['value'] >= 0 and evt['value'] < 128
    evt_type = music21.midi.ChannelVoiceMessages.NOTE_ON if is_on else music21.midi.ChannelVoiceMessages.NOTE_OFF
    me1 = music21.midi.MidiEvent(mt)
    me1.type = evt_type
    me1.pitch = evt['value']
    me1.channel = chan
    me1.velocity = note_vel
    return me1

  for evt in events:
    if evt['type'] == 'ts':
      dt = int(np.floor(evt['value'] * ticks_per_second))

    elif evt['type'] == 'vel':
      assert evt['value'] >= 0 and evt['value'] < 128
      note_vel = evt['value']

    elif evt['type'] == 'on' or evt['type'] == 'off':
      mdt = music21.midi.DeltaTime(mt, dt)
      dt = 0
      me1 = make_note_event(evt, 1, note_vel)
      mt.events.append(mdt)
      mt.events.append(me1)

def write_midi_file(evts, p):
  mf = music21.midi.MidiFile()
  mt = music21.midi.MidiTrack(1)

  to_midi_events(mt, evts)
  mf.tracks.append(mt)
  mf.open(p, 'wb')
  mf.write()
  mf.close()

def parse_performance_event_indices(file):
  inds = np.zeros((file['var'].shape[0],))
  for i in range(len(inds)):
    inds[i] = file['var'][i][0][2][0]
  return inds

def get_device(force_cuda=False):  
  return torch.device('cuda:0' if force_cuda or torch.cuda.is_available() else 'cpu')