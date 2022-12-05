years = { '2018', '2017', '2015', '2014', '2013', '2011', '2009', '2008' };

mid_files = shared_utils.io.find( shared_utils.io.fullfiles(proj_root, 'data/maestro/maestro-v3.0.0', years), '.midi' );
bpm = 120;

dst_p = fullfile( proj_root, 'data/maestro-mat' );

parfor i = 1:numel(mid_files)
  fprintf( '\n %d of %d', i, numel(mid_files) );
  
  dst_file_p = fullfile( dst_p, sprintf('%s.mat', shared_utils.io.filenames(mid_files{i}, false)) );
  if ( exist(dst_file_p, 'file') )
    continue;
  end
  
  try
    mid = read_midi_file( mid_files{i}, bpm );
    do_save( dst_file_p, mid );
  catch err
    warning( err.message );
  end
end

%%

function do_save(p, var)
save( p, 'var' );
end