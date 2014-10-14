function count = edge_code_counts(G, L, params)
% Counts the appearance times of different edge codes.
% Note an edge code is a tuple (s_label, edge_type, t_label)
% where s_label and t_label are starting and ending nodes' labels.

% Reads graph properties.
[e1, e2] = find(G > 0);
code_count = params.code_count;
m = code_count ^ 2;
edge_count = length(e1);
edge_type_count = length(params.edge_types);
edge_type_idx(params.edge_types) = 1 : edge_type_count;

% Counts the appearance times of edge codes.
count = zeros(1, code_count * code_count * edge_type_count);

for i = 1 : edge_count
    ei = edge_type_idx(abs(G(e1(i), e2(i))));  
    id = sub2ind([code_count, code_count], L(e1(i)), L(e2(i))) + (ei - 1) * m;
    count(id) = count(id) + 1;
end
end
