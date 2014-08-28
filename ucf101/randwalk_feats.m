function [node_feats, edge_feats] = randwalk_feats(S, G, ...
    node_weights, edge_weights, edge_code_idx, params)
% Generate feature vectors from subgraphs centered on each graph node.

[node_count, code_count] = size(S);
m = code_count * code_count;
[e1, e2] = find(G > 0);
edge_count = length(e1);
edge_type_count = length(params.edge_types);
edge_type_idx(params.edge_types) = 1 : edge_type_count;

% Computes random walk weights in each subgraph.
if ~exist('node_weights', 'var') || ~exist('edge_weights', 'var')
    [node_weights, edge_weights] = random_walk_weights(...
        G, params.subgraph_radius);
end

% For each node, finds the code words scored higher than threshold.
CI = cell(node_count, 1); 
for i = 1 : node_count
    CI{i} = find(S(i, :) > params.code_score_thresh);
end

% Computes edge scores. ES is #edge_count * #edge_codes.
a = sum(max(S > params.code_score_thresh, [], 1));
c = a * a;
row_idx = zeros(edge_count * c, 1);
col_idx = zeros(edge_count * c, 1);
edge_score = zeros(edge_count * c, 1);
ind = 0;
for i = 1 : edge_count
    ei = edge_type_idx(abs(G(e1(i), e2(i))));  
    [X, Y] = meshgrid(CI{e1(i)}, CI{e2(i)});
    Z = sub2ind([code_count, code_count], X(:), Y(:));
    [~, ia, ib] = intersect(edge_code_idx, Z + (ei - 1) * m);
    if isempty(ia)
        continue;
    end
    idx = ind + [1 : length(ia)];
    col_idx(idx) = ia(:);
    row_idx(idx) = ones(length(idx), 1) * i;
    edge_score(idx) = min(S(e1(i), X(ib)), S(e2(i), Y(ib)));
    ind = ind + length(idx);
end
if ind < length(row_idx)
    row_idx(ind + 1 : end) = [];
    col_idx(ind + 1 : end) = [];
    edge_score(ind + 1 : end) = [];
end
ES = sparse(row_idx, col_idx, edge_score, ...
    edge_count, length(edge_code_idx));

% Generates feature vector for each subgraph.
node_feats = zeros(node_count, code_count);
edge_feats = zeros(node_count, length(edge_code_idx));
for i = 1:node_count
    flg = node_weights(i, :) > 0;
    if sum(flg) < 2
        continue;
    end
    node_feats(i, :) = weighted_pooling(...
        S(flg, :), node_weights(i, flg), params.pooling_mode);
    
    flg2 = edge_weights(i, :) > 0;
    if ~any(flg2)
        continue;
    end
    edge_feats(i, :) = full(weighted_pooling(...
        ES(flg2, :), edge_weights(i, flg2), params.pooling_mode));
end

end

function f = weighted_pooling(feats, weights, mode)
% Performs weighted pooling over a set of features.
% Options for mode:
%     1:    max pooling.
%     2:    sum pooling (average pooling).
feats = bsxfun(@times, feats, weights(:));

if mode == 1
    f = max(feats, [], 1);
elseif mode == 2
    f = sum(feats, 1) / sum(weights);
end

end
