function count = edge_code_counts(G, S, params)
% Counts the appearance times of different edge codes.
% Note an edge code is a tuple (s_label, edge_type, t_label)
% where s_label and t_label are starting and ending nodes' labels.

% Reads graph properties.
[node_count, code_count] = size(S);
[e1, e2] = find(G > 0);
m = code_count ^ 2;
edge_count = length(e1);
edge_type_count = length(params.edge_types);
edge_type_idx(params.edge_types) = 1 : edge_type_count;

% For each node, finds the code words scored higher than threshold.
CI = cell(node_count, 1); 
for i = 1 : node_count
    CI{i} = find(S(i, :) > params.code_score_thresh);
end

% Counts the appearance times of edge codes.
count = zeros(1, code_count * code_count * edge_type_count);
for i = 1 : edge_count
    ei = edge_type_idx(abs(G(e1(i), e2(i))));  
    [X, Y] = meshgrid(CI{e1(i)}, CI{e2(i)});
    Z = sub2ind([code_count, code_count], X(:), Y(:)) + (ei - 1) * m;
    count(Z) = count(Z) + 1;
end

end