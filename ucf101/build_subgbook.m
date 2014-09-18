init_ucf101;

% =========================================================================
%
% Creates subgraph codebook.
%
% =========================================================================
ncls = 101;
D = cell(3, 101);

% Loads data.
for vid = 1:length(video_list)
    label = class_labels(vid);
    if ~exist([randw_feat_path filesep num2str(vid) 'rf.mat'], 'file')
        continue;
    end
    load([randw_feat_path filesep num2str(vid) 'rf.mat']);
    for t = 1:3
        if used_for_training(vid, t) < 1
            continue;
        end 
        X = [cell2mat(node_feats(t, :)) cell2mat(edge_feats(t, :))];
        idx = randsample(size(X, 1), min(size(X, 1), 100));
        D{t, label} = [D{t, label};  normr(X(idx, :))];
    end
end

% Kmeans clustering.
K = 100;
C = cell(t, ncls);
opt = statset('MaxIter', 100, 'Display', 'final');
for t = 1:3
    parfor cid = 1:ncls
        n = size(D{t, cid}, 1);
        sample_idx = randsample(n, min(n, 1e4));
        [~, C{t, cid}] = kmeans(D{t, cid}(sample_idx, :), K, ...
            'replicates', 3, 'emptyaction', 'drop', 'Options', opt);
    end
end

save(['data' filesep 'subgbook.mat'], 'C', '-v7.3');

