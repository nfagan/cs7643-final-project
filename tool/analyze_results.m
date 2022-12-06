model_outs = shared_utils.io.findmat( fullfile(proj_root, 'data/stats') );
model_stats = cellfun( @load, model_outs );

min_loss = cellfun( @min, {model_stats.valid_loss} );
max_acc = cellfun( @max, {model_stats.valid_acc} );

loss_table = table( min_loss(:), max_acc(:), 'VariableNames', {'MinLoss', 'MaxAcc'} );
loss_table.Properties.RowNames = shared_utils.io.filenames( model_outs );

%%  transformer train curves

is_tform = contains( model_outs, 'tform' ) & ~contains( model_outs, 'jsb' );
min_num_eps = min( cellfun(@numel, {model_stats(is_tform).valid_acc}) );
model_names = shared_utils.io.filenames( model_outs(is_tform) );

valid_loss = arrayfun( @(x) x.valid_loss(1:min_num_eps), model_stats(is_tform), 'un', 0 );
train_loss = arrayfun( @(x) x.loss(1:min_num_eps), model_stats(is_tform), 'un', 0 );

losses = [ valid_loss; train_loss ];
kind_labs = [ repmat({'valid'}, numel(valid_loss), 1); repmat({'train'}, numel(train_loss), 1) ];
model_labs = repmat( model_names(:), 2, 1 );

ax = gca; cla( ax ); hold( ax, 'on' );
for i = 1:numel(losses)
  plot( ax, losses{i}, 'displayname', strrep(sprintf('%s-%s', model_labs{i}, kind_labs{i}), '_', ' ') );
end

legend;