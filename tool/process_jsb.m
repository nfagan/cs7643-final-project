% chor_file = 'jsb-chorales-16th.json';
chor_file = 'Jsb16thSeparated.json';
js = jsondecode( fileread(fullfile(proj_root, '/data/JSB-Chorales-dataset', chor_file)) );

bpm = 50;
beats_per_row = 1 / 16;
s_per_row = 1 / (bpm / 60) * beats_per_row;

pe_train = cellfun( @(x) jsb_notes_matrix_to_perf_events(x, s_per_row), js.train, 'un', 0 );
pe_test = cellfun( @(x) jsb_notes_matrix_to_perf_events(x, s_per_row), js.test, 'un', 0 );
pe_valid = cellfun( @(x) jsb_notes_matrix_to_perf_events(x, s_per_row), js.valid, 'un', 0 );

%%

tot_pe = [ pe_train; pe_valid; pe_test ];
for i = 1:numel(tot_pe)
  dst_p = fullfile( proj_root, 'data/jsb-performance-events', sprintf('pe-%d.mat', i) );
  var = tot_pe{i};
  save( dst_p, 'var' );
end