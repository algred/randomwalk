function [node_weights, edge_weights] = random_walk_weights_approx2(G, scale)
% Computes random walk weights for subgraphs that centered on graph nodes. 
% 
% Inputs:
%   G:       symmetric transition matrix.
%   Scale:   radius of subgraphs.
%
% Outputs:
%   node_weights:    n * n matrix, ith row contains random walk weights on
%                    nodes in subgraph centered at node i.
%   edge_weights:    n * m matrix, ith row contains random walk weights on 
%                    edges in the subgraph centered at node i.
%                  
% To compute Compute Time distance between graph nodes, This version uses 
% the algorithm "Clustering and Embedding Using Commute % Times", 
% Huaijun Qiu et. al. PAMI 2007. 

[e1, e2] = find(G > 0); e1 = e1(:); e2 = e2(:);
G = (G ~= 0);
n = size(G, 1);
m = length(e1);

% Computes the distances among graph nodes.
D = graphallshortestpaths(sparse(G));

% Computes the random walk weights in subgraphs centered on each node.
node_val = cell(n, 1);
node_idxx = cell(n, 1);
node_idxy = cell(n, 1);
edge_val = cell(n, 1);
edge_idxx = cell(n, 1);
edge_idxy = cell(n, 1);
fprintf('Computing randomwalk weights...');
fprintf('node num = %d, edge num = %d\n', n, m);
for i = 1:n
    flg = (D(i, :) <= scale);
    idx = find(flg);
    s = sum(flg(1:i));
    subg = G(flg, flg);
    if size(subg, 1) < 2
        continue;
    end
    [num_visits_n, num_steps_n] = randw(subg, s);
    node_val{i} = num_visits_n ./ num_steps_n;
    node_idxx{i} = idx(:);
    node_idxy{i} = ones(length(idx), 1) * i;
    
    % Computes the approximate random walk weights of edges.
    node_map = []; node_map(idx, 1) = 1:length(idx);
    edge_idx = find(ismember(e1, idx) & ismember(e2, idx));
    this_e1 = e1(edge_idx); this_e2 = e2(edge_idx);
    node_degree = sum(subg, 2);
    edge_degree = node_degree(node_map(this_e1)) + ...
        node_degree(node_map(this_e2)) - 2;
    s1 = (this_e1 == i | this_e2 == i);
    num_visits_e = edge_degree / max(edge_degree(s1));
    num_steps_e = max(num_steps_n(node_map(this_e1)), ...
        num_steps_n(node_map(this_e2)));
    edge_val{i} = num_visits_e(:) ./ num_steps_e(:);
    edge_idxx{i} = edge_idx(:);
    edge_idxy{i} = ones(length(edge_idx), 1) * i;
end

node_val = cell2mat(node_val);
node_idxx = cell2mat(node_idxx);
node_idxy = cell2mat(node_idxy);
node_weights = sparse(node_idxy, node_idxx, node_val, n, n);

edge_val = cell2mat(edge_val);
edge_idxx = cell2mat(edge_idxx);
edge_idxy = cell2mat(edge_idxy);
edge_weights = sparse(edge_idxy, edge_idxx, edge_val, n, m);
end

function [N, CT] = randw(G, s)
n = size(G, 1);

% Computes the stationary distribution. p_v = d(v) / 2|E|.
edge_count = sum(sum(triu(G)));
P = sum(G > 0, 2) / (2 * edge_count);

% Expected number of visits to nodes in a roundtrip starting from s.
N = P / P(s);
N(s) = 1;

% Computes Commute Time distance (CT).
% Follows algorithm in "Clustering and Embedding Using Commute Times".
% Second equation on Page 1876, right column.
D = sum(G, 1);
L = diag(D) - G;
vol = sum(D);
[Phi, Lambda] = eig(L);
Lambda = diag(Lambda)';
CT = zeros(n, 1);
for i = 1:n
    if i == s
        CT(s) = 1;
        continue;
    end
    delta = Phi(s, 2:end) - Phi(i, 2:end); 
    CT(i) = vol * sum(delta .* delta ./ Lambda(2:end));
end

end
