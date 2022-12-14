mid_file = fullfile( proj_root, 'data/maestro/maestro-v3.0.0/2018/MIDI-Unprocessed_Recital5-7_MID--AUDIO_05_R1_2018_wav--3.midi' );
bpm = 120;
mid = read_midi_file( mid_file, bpm );

%%

perf_events = gen_perf_events( mid );

%%

name = 'solo_flute';
mid_file = fullfile( proj_root, 'data/my-midi/', sprintf('%s.mid', name) );
mid = read_midi_file( mid_file, bpm );
perf_events = gen_perf_events( mid, true );
do_save( fullfile(fileparts(mid_file), sprintf('%s.mat', name)), perf_events );

%%

mid_files = shared_utils.io.findmat( fullfile(proj_root(), 'data/maestro-mat') );
parfor i = 1:numel(mid_files)
  fprintf( '\n %d of %d', i, numel(mid_files) );
  
  fname = shared_utils.io.filenames( mid_files{i}, true );
  dst_p = fullfile( proj_root, 'data/maestro-performance-events', fname );
  if ( exist(dst_p, 'file') )
    fprintf( ' Skipping, already exists.' );
    continue;
  end
  
  try
    mid = shared_utils.io.fload( mid_files{i} );
    perf_events = gen_perf_events( mid );
    do_save( dst_p, perf_events );
  catch err
    warning( err.message );
  end
end

%%

dst_p = fullfile( proj_root, 'data/jsb-performance-events' );
dst_files = shared_utils.io.find( dst_p, '.mat' );

seq_lens = zeros( numel(dst_files), 1 );
parfor i = 1:numel(dst_files)
  fprintf( '\n %d of %d', i, numel(dst_files) );
  f = shared_utils.io.fload( dst_files{i} );
  seq_lens(i) = numel( f );
end

%%

function do_save(p, var)
save( p, 'var' );
end

