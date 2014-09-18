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
            ~exist([randw_feat_path filesep imname '_rf' sfx '.mat'], 'file')
        continue;
    end
    
    label = class_labels(i);
    
    load([randw_feat_path filesep imname '_rf' sfx '.mat']);
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

opt = statset('UseParallel', true, 'MaxIter', 200, 'Display', 'final');

parfor cid = 1:ncls
    n = size(D{cid}, 1);
    sample_idx{cid} = randsample(n, min(n, 2e4));
    if use_vlfeat
        if fisher_encoding
            [C{cid}.means C{cid}.cov C{cid}.priors] = vl_gmm(...
                D{cid}(sample_idx{cid}, :)', K);
        else
            [C{cid}, ~] = vl_kmeans(D{cid}(sample_idx{cid}, :)', K, ...
                'Initialization', 'plusplus', 'Algorithm', 'Elkan');
            C{cid} = C{cid}';
        end
    else
        [~, C{cid}] = kmeans(D{cid}(sample_idx{cid}, :), K, ...
            'replicates', 12, 'emptyaction', 'drop', 'Options', opt);
    end
    if ~fisher_encoding
        a = max(isnan(C{cid}), [], 2);
        C{cid} = C{cid}(a < 1, :);
        b = sqrt(sum(C{cid} .* C{cid}, 2));
        C{cid} = C{cid}(b > 0.8 & b <= 1, :);
    end
end

save(['data' filesep codebook_name '.mat'], 'C');

