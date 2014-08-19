function [node_feats, edge_feats, node_weights, edge_weights] = randwalk_feats(RS, PS, G,  params)
% Generate feature vectors from subgraphs centered on each graph node.

[n1, cr] = size(RS);
[n2, cp] = size(PS);
node_count = n1 + n2;
code_count = cr + cp;
m = code_count * code_count;
[e1, e2] = find(G > 0);
edge_count = length(e1);

% Computes random walk weights in each subgraph.
[node_weights, edge_weights] = random_walk_weights(...
    G, params.subgraph_radius);

% Thresholds the code word scores to save space and speedup.
ridx = find(max(RS > params.code_score_thresh, [], 1));
pidx = find(max(PS > params.code_score_thresh, [], 1));
cr1 = length(ridx); cp1 = length(pidx);
code_count1 = cr1 + cp1;
m1 = code_count1 * code_count1;
[X, Y] = meshgrid([ridx pidx + cr]);
code_idx1 = sub2ind([code_count, code_count], X(:), Y(:));
code_idx1 = code_idx1(:);
code_idx = [];
for i = 1 : params.edge_type_count
    code_idx = [code_idx; code_idx1 + (i - 1) * m];
end

% Computes edge scores.
col_idx = zeros(edge_count * m1, 1);
row_idx = zeros(edge_count * m1, 1);
val = zeros(edge_count * m1, 1);
ind = 0;
for i = 1 : edge_count
    if e1(i) > n1 && e2(i) > n1
        [X, Y] = meshgrid((cr1 + 1) : (cr1 + cp1));
        [X1, Y1] = meshgrid(pidx);
        val1 = min(PS(e1(i) - n1, X1(:)), PS(e2(i) - n1, Y1(:)));     
    elseif e1(i) <= n1 && e2(i) <= n1
        [X, Y] = meshgrid(1 : cr1);
        [X1, Y1] = meshgrid(ridx);
        val1 = min(RS(e1(i), X1(:)), RS(e2(i), Y1(:)));
    elseif e1(i) <= n1 && e2(i) > n1
        [X, Y] = meshgrid(1 : cr1, (cr1 + 1) : (cr1 + cp1));
        [X1, Y1] = meshgrid(ridx, pidx);
        val1 = min(RS(e1(i), X1(:)), PS(e2(i) - n1, Y1(:)));
    elseif e1(i) > n1 && e2(i) <= n1
        [X, Y] = meshgrid((cr1 + 1) : (cr1 + cp1), 1 : cr1);
        [X1, Y1] = meshgrid(pidx, ridx);
        val1 = min(PS(e1(i) - n1, X1(:)), RS(e2(i), Y1(:)));
    end
    
    idx1 = sub2ind([code_count1, code_count1], X(:), Y(:));
    edge_type = abs(G(e1(i), e2(i)));
    offset = (edge_type - 1) * m1;
    col_idx(ind + [1 : length(idx1)]) = idx1 + offset;
    row_idx(ind + [1 : length(idx1)]) = ones(length(idx1), 1) * i;
    val(ind + [1 : length(idx1)]) = val1;
    ind = ind + length(idx1);
end
if ind < length(row_idx)
    row_idx(ind + 1 : end) = [];
    col_idx(ind + 1 : end) = [];
    val(ind + 1 : end) = [];
end
ES = sparse(col_idx, row_idx, val, m1*params.edge_type_count, edge_count);

% Generates feature vector for each subgraph.
M = node_weights > 0;
ME = edge_weights > 0;
node_feats = zeros(node_count, code_count);
row_idx = zeros(node_count * 10 * m1, 1);
col_idx = row_idx; val = row_idx;
ind = 0;
for i = 1 : node_count
    if any(M(i, 1 : n1))
        node_feats(i, 1 : cr) = weighted_pooling(...
            RS(M(i, 1 : n1), :), ... 
            node_weights(i, M(i, 1 : n1)), ...
            params.pooling_mode);
    end
    
    if any(M(i, (n1 + 1) : n2))
        node_feats(i, (cr + 1) : end) = weighted_pooling(...
            PS(M(i, (n1 + 1) : n2), :), ...
            node_weights(i, M(i, (n1 + 1) : n2)), ...
            params.pooling_mode);
    end
    
    val1 = weighted_pooling(...
        ES(:, ME(i, :))', edge_weights(i, ME(i, :)), params.pooling_mode);
    val1 = full(val1);
    col_idx1 = code_idx(val1 ~= 0);
    row_idx1 = ones(length(col_idx1), 1) * i;
    val1 = val1(val1 ~= 0);

    if ind + length(row_idx1) > length(row_idx)
        row_idx = [row_idx; zeros(node_count * 10 * m1, 1)];
        col_idx = [col_idx; zeros(node_count * 10 * m1, 1)];
        val = [val; zeros(node_count * 10 * m1, 1)];
    end
    row_idx(ind + 1 : ind + length(row_idx1)) = row_idx1;
    col_idx(ind + 1 : ind + length(col_idx1)) = col_idx1;
    val(ind + 1 : ind + length(val1)) = val1;
    ind = ind + length(row_idx1);
end
if ind < length(row_idx)
    row_idx(ind + 1 : end) = [];
    col_idx(ind + 1 : end) = [];
    val(ind + 1 : end) = [];
end
edge_feats = sparse(row_idx, col_idx, val, ...
    node_count, m * params.edge_type_count);
end

function f = weighted_pooling(feats, weights, mode)
% Performs weighted pooling over a set of features.
% Options for mode:
%     1:    max pooling.
%     2:    sum pooling (average pooling).
feats = feats .* repmat(weights(:), 1, size(feats, 2));

if mode == 1
    f = max(feats, [], 1);
elseif mode == 2
    f = sum(feats, 1);
end

end