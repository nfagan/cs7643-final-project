function pe = jsb_notes_matrix_to_perf_events(m, s_per_row)

on_notes = nan( 1, size(m, 2) );
stream = [];

for j = 1:size(m, 1)
  mr = m(j, :);
  mr = max( mr, zeros(size(mr)) );

  within_tstep = 0;
  within_tstep_size = 1e-2;
  t0 = (j - 1) * s_per_row;

  for k = 1:numel(mr)
    if ( on_notes(k) == mr(k) )
      continue % still playing
    end

    if ( ~isnan(on_notes(k)) )
      % another note is already on, so stop playing it first.
      stream(end+1, :) = [ 0, on_notes(k), t0 + within_tstep ];
      within_tstep = within_tstep + within_tstep_size;
      on_notes(k) = nan;
    end

    % this is a new note. first check whether another voice is already
    % playing this note, and make it stop playing that note if so.
    hit_cond = false;
    for h = 1:numel(mr)
      if ( h ~= k && on_notes(h) == mr(k) )
        assert( ~hit_cond );
        hit_cond = true;
        % another voice is already playing the note we want to play.
        stream(end+1, :) = [ 0, on_notes(h), t0 + within_tstep ];
        on_notes(h) = nan;
        within_tstep = within_tstep + within_tstep_size;
      end
    end

    stream(end+1, :) = [ 1, mr(k), t0 + within_tstep ];
    on_notes(k) = mr(k);
    within_tstep = within_tstep + within_tstep_size;
  end
end

% Terminate remaining notes.
within_tstep = 0;
within_tstep_size = 1e-2;
t0 = size(m, 1) * s_per_row;
for j = 1:numel(on_notes)
  if ( ~isnan(on_notes(j)) )
    stream(end+1, :) = [ 0, on_notes(j), t0 + within_tstep ];
    on_notes(j) = nan;
    within_tstep = within_tstep + within_tstep_size;
  end
end

s = to_midi_struct( stream );

if ( 1 )
  validate_midi_struct( s );
end

pe = gen_perf_events( s, true );

end

function validate_midi_struct(s)

is_on = false( 1, 128 );
for i = 1:numel(s)
  ni = s(i).Note;
  assert( ni >= 0 && ni < 128 );
  if ( is_on(ni+1) )
    assert( s(i).Type == "NoteOff" );
    is_on(ni+1) = false;
  else
    assert( s(i).Type == "NoteOn" );
    is_on(ni+1) = true;
  end
end

assert( ~any(is_on) );

end

function s = to_midi_struct(stream)

types = { "NoteOff", "NoteOn" };
types = types(stream(:, 1)+1);
timestamps = arrayfun( @(x) x, stream(:, 3), 'un', 0 );
notes = arrayfun( @(x) x, stream(:, 2), 'un', 0 );
vels = arrayfun( @(x) x, repmat(127, size(types)), 'un', 0 );

s = struct( 'Type', types(:)', 'Timestamp', timestamps', 'Note', notes', 'Velocity', vels(:)' );

end