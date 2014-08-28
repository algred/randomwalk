function [node_feats, edge_feats] = randwalk_feats(S, L, G, ...
    node_weights, edge_weights, edge_code_idx, params)
% Generate feature vectors from subgraphs centered on each graph node.
node_count = length(S);
code_count = params.code_count;
[e1, e2] = find(G > 0);
edge_count = length(e1);
edge_type_count = length(params.edge_types);
edge_type_idx(params.edge_types) = 1 : edge_type_count;
m = code_count * code_count;

% Computes random walk weights in each subgraph.
if ~exist('node_weights', 'var') || ~exist('edge_weights', 'var')
    [node_weights, edge_weights] = random_walk_weights(...
        G, params.subgraph_radius);
end

% Computes edge scores.
ES = zeros(edge_count, 2);
for i = 1 : edge_count
    ei = edge_type_idx(abs(G(e1(i), e2(i))));  
    edge_code = sub2ind([code_count, code_count], L(e1(i)), L(e2(i))) + ...
        (ei - 1) * m;
    id = find(edge_code_idx == edge_code);
    if isempty(id)
        continue;
    end
    ES(i, :) = [double(id) min(S(e1(i)), S(e2(i)))];
end

% Generates feature vector for each subgraph.
node_feats = zeros(node_count, code_count);
edge_feats = zeros(node_count, length(edge_code_idx));
edge_flg = ES(:, 1)' > 0;
for i = 1:node_count
    flg = node_weights(i, :) > 0;
    if sum(flg) < 2
        continue;
    end
    node_feats(i, :) = weighted_pooling([L(flg) S(flg)], ...
        node_weights(i, flg), code_count, params.pooling_mode);
    
    flg2 = (edge_weights(i, :) > 0) & edge_flg;
    if ~any(flg2)
        continue;
    end
    edge_feats(i, :) = full(weighted_pooling(ES(flg2, :), ...
        edge_weights(i, flg2), length(edge_code_idx), params.pooling_mode));
end

end

function f = weighted_pooling(LS, weights, code_count, mode)
% Performs weighted pooling over a set of features.
% Options for mode:
%     1:    max pooling.
%     2:    sum pooling (average pooling).
f = zeros(1, code_count);
if mode == 1
    v = accumarray(LS(:, 1), LS(:, 2) .* weights(:), [], @max);
    f(1:length(v)) = v';
elseif mode == 2
    v = accumarray(LS(:, 1), LS(:, 2) .* weights(:), [], @sum);
    f(1:length(v)) = v';
    f = f / sum(weights);
end

end
