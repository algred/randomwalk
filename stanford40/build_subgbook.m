init_stanford40;

% =========================================================================
%
% Creates subgraph codebook.
%
% =========================================================================
ncls = 40;
D = cell(ncls, 1);

% Loads data.
for i = 1:length(annotation)
    [~, imname, ext] = fileparts(annotation{i}.imageName);
    
    if used_for_training(i) < 1 || ...
            ~exist([randw_feat_path filesep imname '_rf.mat'], 'file')
        continue;
    end
    
    label = class_labels(i);
    
    load([randw_feat_path filesep imname '_rf.mat']);
    X = [node_feats edge_feats];
    X = X ./ (repmat(sqrt(sum(X .* X, 2)), 1, size(X, 2)) + eps);
    s = sum(X > 0, 2);
    [~, ix] = sort(s, 'descend');
    D{label} = [D{label}; X(ix(1:min(length(ix), 50)), :)];
end

% Kmeans clustering.
K = 100;
C = cell(ncls, 1);
sample_idx = cell(ncls, 1);
c_idx = cell(ncls, 1);
opt = statset('UseParallel', true, 'MaxIter', 200, 'Display', 'final');
for cid = 1:ncls
    n = size(D{cid}, 1);
    sample_idx{cid} = randsample(n, min(n, 1e4));
    [c_idx{cid}, C{cid}] = kmeans(D{cid}(sample_idx{cid}, :), K, ...
        'replicates', 12, 'emptyaction', 'drop', 'Options', opt);
end

save(['data' filesep 'subgbook.mat'], 'C');

