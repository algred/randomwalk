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
    
    load([randw_feat_path filesep num2str(vid) '_rf' sfx '.mat']);
    X = [cell2mat(node_feats') cell2mat(edge_feats')];
    sample_idx = randsample(size(X, 1), min(size(X, 1), 200));
    X = X(sample_idx, :);
    X = X ./ repmat(sqrt(sum(X .* X, 2)) + eps, 1, size(X, 2));
    D{label} = [D{label}; X];
    
    load([randw_feat_path filesep num2str(vid) '_rf' sfx '_flip.mat']);
    X = [cell2mat(node_feats') cell2mat(edge_feats')];
    sample_idx = randsample(size(X, 1), min(size(X, 1), 200));
    X = X(sample_idx, :);
    X = X ./ repmat(sqrt(sum(X .* X, 2)) + eps, 1, size(X, 2));
    D{label} = [D{label}; X];
end

% Kmeans clustering.
K = 100;
C = cell(ncls, 1);
sample_idx = cell(ncls, 1);

opt = statset('UseParallel', true, 'MaxIter', 200, 'Display', 'final');
for cid = 1:ncls
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

