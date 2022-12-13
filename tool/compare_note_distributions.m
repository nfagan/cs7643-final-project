%%  rpr transformer generated

bpm = 120;

skip_invalid = true;

seqs = 0:3;

note_nums = [];
note_vels = [];
for i = 1:numel(seqs)
  mid_fname = sprintf( 'rpr-transformer-checkpoint.pth-seq-%d-gen.mid', seqs(i) );
  mid_file = fullfile( proj_root, 'data/my-midi/', mid_fname );
  mid = read_midi_file( mid_file, bpm );
  [~, note_info] = gen_perf_events( mid, true, skip_invalid );

  note_nums = [ note_nums; note_info(:, 3) ];
  note_vels = [ note_vels; note_info(:, 4) ];
end

%%  real

real_note_nums = [];
real_note_vels = [];

rng( 2337 );
real_mid_files = shared_utils.io.find( fullfile(proj_root, 'data/maestro/maestro-v3.0.0'), '.midi', true );
real_mid_files = real_mid_files(randperm(numel(real_mid_files), 8))';

for i = 1:numel(real_mid_files)
  [~, note_info] = gen_perf_events( read_midi_file( real_mid_files{i}, bpm ) );
  real_note_nums = [ real_note_nums; note_info(:, 3) ];
  real_note_vels = [ real_note_vels; note_info(:, 4) ];
end

%%  compare

ax1 = cla( subplot(1, 2, 1) );
ax2 = cla( subplot(1, 2, 2) );

plot_comparison( ax1, real_note_nums, real_note_vels );
plot_comparison( ax2, note_nums, note_vels );

title( ax1, 'Real sequences' );
title( ax2, 'Unconditioned generated sequences' );

ylabel( ax1, 'Note Velocity' );
xlabel( ax1, 'Note Pitch (MIDI note number)' );

%%

function plot_comparison(ax, note_nums, note_vels)

assert( numel(note_nums) == numel(note_vels) );
num_plot = min( 5e3, numel(note_nums) );

hold( ax, 'on' );
xlim( ax, [30, 105] );
ylim( ax, [0, 32] );
axis( ax, 'square' );

scatter( ax, note_nums(1:num_plot), note_vels(1:num_plot), 4 );
lims = get( ax, 'xlim' );

colors = { 'r', 'g', 'b' };
for i = 3
  p = polyfit( note_nums, note_vels, i );
  x = linspace( lims(1), lims(2) );
  y = polyval( p, x );
  plot( ax, x, y, colors{i} );
end

end