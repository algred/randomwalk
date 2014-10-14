function [node_feats, edge_feats] = randwalk_feats2(S, L, G, ...
    node_weights, edge_weights, params)

S = double(S);
L = double(L);
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
es1 = zeros(edge_count, 2);
es2 = zeros(edge_count, 2);
for i = 1 : edge_count
    edge_type = edge_type_idx(abs(G(e1(i), e2(i))));  
    edge_code1 = (L(e1(i)) - 1) * edge_type_count + edge_type;
    edge_code2 = (L(e2(i)) - 1) * edge_type_count + edge_type;
    es1(i, :) = [edge_code1 S(e1(i))];
    es2(i, :) = [edge_code2 S(e2(i))];
end

% Generates feature vector for each subgraph.
node_feats = zeros(node_count, code_count);
edge_code_count = code_count * edge_type_count;
edge_feats = zeros(node_count, edge_code_count);
edge_flg1 = es1(:, 1)' > 0;
edge_flg2 = es2(:, 1)' > 0;
for i = 1:node_count
    flg = node_weights(i, :) > 0;
    if sum(flg) < 2
        continue;
    end
    node_feats(i, :) = weighted_pooling([L(flg) S(flg)], ...
        node_weights(i, flg), code_count, params.pooling_mode);
    
    flg_es1 = (edge_weights(i, :) > 0) & edge_flg1;
    flg_es2 = (edge_weights(i, :) > 0) & edge_flg1;
    if ~any(flg_es1 | flg_es2)
        continue;
    end
    f = zeros(1, edge_code_count * 2);
    if any(flg_es1)
        f(1 : edge_code_count) = full(weighted_pooling(es1(flg_es1, :), ...
        edge_weights(i, flg_es1), edge_code_count, params.pooling_mode));
    end
    if any(flg_es2)
        f(edge_code_count + 1 : end) = full(weighted_pooling(es1(flg_es2, :), ...
        edge_weights(i, flg_es2), edge_code_count, params.pooling_mode));
    end
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
