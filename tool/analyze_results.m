model_outs = shared_utils.io.findmat( fullfile(proj_root, 'data/stats') );
model_stats = cellfun( @load, model_outs );

min_loss = cellfun( @min, {model_stats.valid_loss} );
max_acc = cellfun( @max, {model_stats.valid_acc} );
last_loss = cellfun( @(x) x(end), {model_stats.valid_loss} );
mean_loss_last4 = cellfun( @(x) mean(x(end-3:end)), {model_stats.valid_loss} );

loss_table = table( min_loss(:), last_loss(:), mean_loss_last4(:), max_acc(:) ...
  , 'VariableNames', {'MinLoss', 'LastLoss', 'MeanLossLast4', 'MaxAcc'} );
loss_table.Properties.RowNames = shared_utils.io.filenames( model_outs );

%%  transformer train curves

is_tform = contains( model_outs, 'perf' ) & ~contains( model_outs, 'jsb' );
is_tform = contains( model_outs, 'jsb' );
min_num_eps = min( cellfun(@numel, {model_stats(is_tform).valid_acc}) );
model_names = shared_utils.io.filenames( model_outs(is_tform) );

valid_loss = arrayfun( @(x) x.valid_loss, model_stats(is_tform), 'un', 0 );
train_loss = arrayfun( @(x) x.loss, model_stats(is_tform), 'un', 0 );

clf;
axs = plots.panels( numel(model_names), true );

for i = 1:numel(axs)
  tloss = train_loss{i};
  vloss = valid_loss{i};
  
  ax = axs(i); hold( ax, 'on' );
  h0 = plot( ax, tloss, 'displayname', 'train' );
  h1 = plot( ax, vloss, 'displayname', 'validation' );

  xlim( ax, [0, min(120, numel(tloss))] );
  legend( [h0, h1] );
  title( ax, strrep(model_names{i}, '_', ' ') );
  xlabel( ax, 'Epoch' );
  ylabel( ax, 'Mean loss (negative log-likelihood)' );
  shared_utils.plot.prevent_legend_autoupdate;
  shared_utils.plot.add_horizontal_lines( ax, 2.1 );
end

shared_utils.plot.match_ylims( axs );