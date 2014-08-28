init_high5;

% =========================================================================
%
% Creates subgraph codebook.
%
% =========================================================================
ncls = 4;
D = cell(4, 1);

% Loads data.
for vid = 1:length(video_list)
    if used_for_training(vid) < 1
        continue;
    end
    label = class_labels(vid);
    if label > ncls
        continue;
    end
    
    load([randw_feat_path filesep num2str(vid) '_rf.mat']);
    D{label} = [D{label}; ...
        normr([cell2mat(node_feats') cell2mat(edge_feats')])];
    
    load([randw_feat_path filesep num2str(vid) '_rf_flip.mat']);
    D{label} = [D{label}; ...
        normr([cell2mat(node_feats') cell2mat(edge_feats')])];
end

% Kmeans clustering.
K = 1000;
C = cell(ncls, 1);
sample_idx = cell(ncls, 1);
c_idx = cell(ncls, 1);
opt = statset('UseParallel', true, 'MaxIter', 200, 'Display', 'final');
for cid = 1:ncls
    n = size(D{cid}, 1);
    sample_idx{cid} = randsample(n, min(n, 2e4));
    [c_idx{cid}, C{cid}] = kmeans(D{cid}(sample_idx{cid}, :), K, ...
        'replicates', 12, 'emptyaction', 'drop', 'Options', opt);
end

save(['data' filesep 'subgbook.mat'], 'C');

