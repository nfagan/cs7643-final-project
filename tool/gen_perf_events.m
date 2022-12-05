function perf_events = gen_perf_events(mid, look_for_note_off)

if ( nargin < 2 ), look_for_note_off = false; end

mid_types = [ mid.Type ];
mid_ts = [ mid.Timestamp ];
mid_cc_values = nan( size(mid_types) );
mid_cc_numbers = nan( size(mid_types) );
mid_notes = nan( size(mid_types) );
mid_vels = nan( size(mid_types) );

is_cc = mid_types == 'ControlChange';
is_note = mid_types == 'NoteOn' | mid_types == 'NoteOff';

for i = 1:numel(mid_types)
  if ( is_cc(i) )
    mid_cc_values(i) = mid(i).CCValue;
    mid_cc_numbers(i) = mid(i).CCNumber;
  end
  if ( is_note(i) )
    mid_notes(i) = mid(i).Note;
    mid_vels(i) = mid(i).Velocity;
  end
end

%%  quantize times and velocities

ts_interval = 1e-2;
quant_ts = round( mid_ts, 2 );
min_dur = ts_interval;
max_dur = 1;
num_vel_bins = 32;

ts_bins = min_dur:ts_interval:max_dur;
num_time_bins = numel( ts_bins );
assert( num_time_bins == 100 );

num_notes = 128;

%%  gather note on intervals

is_sustain = mid_cc_numbers == 64 & mid_cc_values >= 64;
[sustain_beg, sustain_dur] = shared_utils.logical.find_islands( is_sustain );
sustain_end = sustain_beg + sustain_dur - 1;
sustain_beg_ts = quant_ts(sustain_beg);
sustain_end_ts = quant_ts(sustain_end);

seq = 1:numel(mid_vels);

is_off = mid_types == 'NoteOn' & mid_vels == 0;
is_on = mid_types == 'NoteOn' & mid_vels > 0;
note_on_inds = find( is_on );

if ( look_for_note_off )
  is_off = is_off | mid_types == 'NoteOff';
end

note_info = [];

for i = 1:numel(note_on_inds)
  ni = note_on_inds(i);
  
  vel = mid_vels(ni);
  vel = floor( vel/128 * num_vel_bins );
  
  note_num = mid_notes(ni);
  
  off_ind = find( is_off & mid_notes == note_num & seq > ni, 1 );
  try
    assert( numel(off_ind) == 1 );
  catch err
    throw( err );
  end
  
  next_on_ind = find( is_on & mid_notes == note_num & seq > ni, 1 );
  
  note_on_t = quant_ts(ni);
  note_off_t = quant_ts(off_ind);
  true_note_dur = max( min_dur, note_off_t - note_on_t );
  
  within_sustain = note_on_t >= sustain_beg_ts & note_on_t < sustain_end_ts;
  if ( sum(within_sustain) > 0 && ~isempty(next_on_ind) )
    next_on_t = quant_ts(next_on_ind);   
    sustain_note_dur = next_on_t - note_on_t;
    true_note_dur = max( true_note_dur, sustain_note_dur );
  end
  
  note_off_t = note_on_t + true_note_dur;
  
  note_info(end+1, :) = [note_on_t, note_off_t, note_num, vel];
end

%%  convert to perf event representation

unique_ts = uniquetol( [note_info(:, 1); note_info(:, 2)] );
last_vel = nan;

perf_events = struct( 'type', {}, 'value', {}, 'index', {} );

% [time_shift (0), note_on, note_off, <vel>]
index_offsets = [0, cumsum([num_time_bins, num_notes, num_notes])];

for i = 1:numel(unique_ts)
  t = unique_ts(i);
  offs = find( note_info(:, 2) == t );
  ons = find( note_info(:, 1) == t );
  
  if ( i > 1 )
    % time shift
    last_t = unique_ts(i-1);
    ts = min( max_dur, t - last_t );
    [~, bi] = ismembertol( ts, ts_bins );
    assert( bi > 0 );    
    bi = bi - 1;  % 0 based index
    perf_events(end+1, 1) = struct( 'type', 'ts', 'value', ts, 'index', index_offsets(1) + bi );
  end
  
  for j = 1:numel(offs)
    nn = note_info(offs(j), 3);
    assert( nn >= 0 && nn < 128 );
    perf_events(end+1, 1) = struct( 'type', 'off', 'value', nn, 'index', index_offsets(3) + nn );
  end
  
  for j = 1:numel(ons)
    nn = note_info(ons(j), 3);
    assert( nn >= 0 && nn < 128 );
    
    vel = note_info(ons(j), 4);
    assert( vel >= 0 && vel < num_vel_bins );
    
    if ( vel ~= last_vel )
      perf_events(end+1, 1) = struct( 'type', 'vel', 'value', vel, 'index', index_offsets(4) + vel );
      last_vel = vel;
    end
    perf_events(end+1, 1) = struct( 'type', 'on', 'value', nn, 'index', index_offsets(2) + nn );
  end
end

%%  check for distinct indices

perf_types = {perf_events.type};
perf_indices = [perf_events.index];
un_types = unique( perf_types );
for i = 1:numel(un_types)
  curr_type = un_types{i};
  rest_types = setdiff( un_types, curr_type );
  curr_i = perf_indices(strcmp(perf_types, curr_type));
  for j = 1:numel(rest_types)
    rest_i = perf_indices(strcmp(perf_types, rest_types{j}));
    assert( isempty(intersect(rest_i, curr_i)) );
  end
end

end