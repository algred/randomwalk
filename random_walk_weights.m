function [node_weights, edge_weights] = random_walk_weights(G, scale)
% WITH BUG AND NOT FINISHED.
%
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

[e1, e2] = find(G > 0); e1 = e1(:); e2 = e2(:);
G = (G ~= 0);
node_degree = sum(G, 2);
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
% fprintf('Computing randomwalk weights...');
% fprintf('node num = %d, edge num = %d\n', n, m);
for i = 1:n
    flg = (D(i, :) <= scale);
    idx = find(flg);
    s = sum(flg(1:i));
    node_val{i} = randw(G(flg, flg), s);
    node_idxx{i} = idx(:);
    node_idxy{i} = ones(length(idx), 1) * i;
    
    node_map = []; node_map(idx, 1) = 1:length(idx);
    edge_idx = find(ismember(e1, idx) & ismember(e2, idx));
    this_e1 = e1(edge_idx); this_e2 = e2(edge_idx);
    edge_degree = node_degree(this_e1) + node_degree(this_e1) - 2;
    s1 = (this_e1 == i | this_e2 == i);
    num_visits = edge_degree / max(edge_degree(s1));
    num_steps = max(node_val{i}(node_map(this_e1)), ...
        node_val{i}(node_map(this_e2)));
    edge_val{i} = num_visits ./ num_steps;
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

function W = randw(G, s)
% Computes the weights of the nodes in the graph centering on node s.
G = G ./ repmat(sum(G, 2), 1, size(G, 2));

% Computes the stationary distribution. p_v = d(v) / 2|E|.
edge_count = sum(sum(triu(G)));
P = sum(G > 0, 2) / (2 * edge_count);

% Computes the fundamental matrix.
Z = inv((eye(size(G)) - G + repmat(P', size(G, 1), 1))); 

% Expected number of visits to nodes in a roundtrip starting from s.
N = P / P(s);
N(s) = 1;

% Expected steps from s to other nodes.
D = (1 ./ P) .* (diag(Z) - Z(s, :)');
D(s) = 1 / P(s);

% Random weights 
W = N ./ D;
end