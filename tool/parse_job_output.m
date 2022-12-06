function outs = parse_job_output(out)

lines = strsplit( out, newline );

has_loss = contains( lines, '; Loss:' );
loss_line = lines{find(has_loss, 1)};
words = strsplit( loss_line, ' ' );
of_ind = find( strcmp(words, 'of'), 1 );
batches_per_epoch = str2double( rem_semi(words{of_ind+1}) );
assert( ~isnan(batches_per_epoch) );
train_lines = lines(has_loss);
ep_start = find( cellfun(@(x) ~isempty(x) && x == 1, strfind(train_lines, sprintf('1 of %d', batches_per_epoch))) );
ep_end = find( contains(train_lines, sprintf('%d of %d', batches_per_epoch)) );
assert( numel(ep_start) == numel(ep_end) && all(ep_end > ep_start) );

batch_stats = parse_stats( train_lines );
assert( ~any(isnan(batch_stats), 'all') );

valid_ind = find( contains(lines, 'Valid Loss') );
assert( numel(valid_ind) == numel(ep_start) );
valid_stats = parse_stats( lines(valid_ind) );

batch_means = nan( numel(ep_start), 2 );
for i = 1:numel(ep_start)
  batch_means(i, :) = mean( batch_stats(ep_start(i):ep_end(i), :), 1 );
end

outs = struct();
outs.loss = batch_means(:, 1);
outs.acc = batch_means(:, 2);
outs.valid_loss = valid_stats(:, 1);
outs.valid_acc = valid_stats(:, 2);
outs.best_acc = max( valid_stats(:, 2) );

end

function stats = parse_stats(lines)

stats = nan( numel(lines), 2 );
for i = 1:numel(lines)
  words = strsplit( lines{i}, ' ' );
  loss_ind = find( strcmpi(words, 'loss:') );
  acc_ind = find( strcmpi(words, 'acc:') );
  stats(i, :) = [str2double(rem_semi(words{loss_ind+1})), str2double(rem_semi(words{acc_ind+1}))];
end

end

function y = rem_semi(x)
y = strrep(x, ';', '');
end