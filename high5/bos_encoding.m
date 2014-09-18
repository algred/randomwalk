init_high5;

% =========================================================================
%
% Encodes videos as Bag-of-Subgraph representation.
%
% =========================================================================
ncls = 4;
D = cell(4, 1);

% Loads data.
% load(['data' filesep 'subgbook.mat']);
load(['data' filesep codebook_name '.mat']);
if ~fisher_encoding
    C = cell2mat(C);
end

% Encoding.
num_video = length(video_list);
n = length(video_list) + sum(used_for_training);
info = zeros(n, 2);
ind = 1;
for vid = 1:num_video
    if exist([randw_feat_path filesep num2str(vid) '_rf' sfx '.mat'], 'file')
        info(ind, :) = [vid, 0];
        ind = ind + 1;
    end

    if exist([randw_feat_path filesep num2str(vid) '_rf' sfx '_flip.mat'], 'file')
        info(ind, :) = [vid, 1];
        ind = ind + 1;
    end
end
if ind < n
    info(ind + 1, :) = [];
    n = size(info, 1);
end

if fisher_encoding
    F2 = cell(1, n);
else
    F = zeros(n, size(C, 1));
end
parfor i = 1:n
    if info(i, 2) > 0
        fname = [randw_feat_path filesep num2str(info(i, 1)) '_rf' sfx '_flip.mat'];
    else
        fname = [randw_feat_path filesep num2str(info(i, 1)) '_rf' sfx '.mat'];
    end
    rf = load(fname);
    X = normr([cell2mat(rf.node_feats') cell2mat(rf.edge_feats')]);
    if fisher_encoding
        F1 = cell(length(C), 1);
        for j = 1:length(C)
            F1{j} = vl_fisher(X', C{j}.means, C{j}.cov, C{j}.priors, ...
                'Fast', 'Normalized');
        end
        F2{i} = cell2mat(F1)';
    else
        F(i, :) = min(pdist2(X, C), [], 1);
    end
end

if fisher_encoding
    F = cell2mat(F2');
end

save(['data' filesep encoding_name '.mat'], 'F', 'info', '-v7.3');

