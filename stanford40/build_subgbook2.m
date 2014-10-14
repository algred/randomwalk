init_stanford40;

% =========================================================================
%
% Creates subgraph codebook.
%
% =========================================================================
ncls = 40;
D = cell(ncls, 1);

% Loads data.
sfx = '_maxpool';
for i = 1:length(annotation)
    [~, imname, ext] = fileparts(annotation{i}.imageName);
    if used_for_training(i) < 1 || ...
            ~exist([randw_feat_path filesep imname '_rf' sfx '.mat'], 'file')
        continue;
    end
    
    label = class_labels(i);
    load([randw_feat_path filesep imname '_rf' sfx '.mat']);
    X = [node_feats edge_feats];
    X = X ./ (repmat(sqrt(sum(X .* X, 2)), 1, size(X, 2)) + eps);
    idx = randsample(size(X, 1), min(size(X, 1), 50));
    D{label} = [D{label}; X(idx, :)];
end

% Kmeans clustering.
K = 100;
C = cell(ncls, 1);
parfor cid = 1:ncls
    n = size(D{cid}, 1);
    sample_idx = randsample(n, min(n, 2e4));
    [C1, ~] = vl_kmeans(D{cid}(sample_idx, :)', K);
    C{cid} = C1';
end

save([randw_feat_path filesep 'bos' sfx '.mat'], 'C');

