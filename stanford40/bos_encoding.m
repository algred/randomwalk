init_stanford40;

% =========================================================================
%
% Encodes each image as bag-of-subgraphs.
%
% =========================================================================

% Loads data.
sfx = '_maxpool';
load([randw_feat_path filesep 'bos' sfx '.mat']);
C = cell2mat(C);

% Encoding.
F = zeros(length(annotation), size(C, 1));
parfor i = 1:length(annotation)
    [~, imname, ext] = fileparts(annotation{i}.imageName);
    if ~exist([randw_feat_path filesep imname '_rf' sfx '.mat'], 'file')
        continue;
    end

    rf = load([randw_feat_path filesep imname '_rf' sfx '.mat']);
    X = [rf.node_feats rf.edge_feats];

    if isempty(X)
        fprintf('No subgraphs, vid = %d. \n', i);
        continue;
    end

    X = X ./ (repmat(sqrt(sum(X .* X, 2)), 1, size(X, 2)) + eps);
    F(i, :) = min(pdist2(X, C), [], 1);
end

save([randw_feat_path filesep 'bos_encoding' sfx], 'F');

