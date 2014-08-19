function [node_weights, edge_weights] = random_walk_weights(G, scale)
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

[e1, e2] = find(G > 0);
G = (G ~= 0);
n = size(G, 1);
m = length(e1);
% node_weights = cell(n, 1);
% edge_weights = cell(n, 1);
node_weights = zeros(n, 1);
edge_weights = zeros(n, 1);

% Computes the due graph.
G2 = zeros(size(e1, 1));
for i = 1:length(e1)
    G2(i, e1 == e1(i) | e2 == e1(i) | e1 == e2(i) | e2 == e2(i)) = 1;
end
G2 = G2 - eye(size(G2));

% Computes the distances among graph nodes.
D = graphallshortestpaths(sparse(G));

% Computes the random walk weights in subgraphs centered on each node.
try
parfor i = 1:n
    flg = (D(i, :) <= scale);
    idx = find(flg);
    s = sum(flg(1:i));
    w = zeros(1, n);
    w(flg) = randw(G(flg, flg), s);
    node_weights(i, :) = w;
    
    edge_flg = (ismember(e1, idx) & ismember(e2, idx));
    if any(edge_flg)
        es = find(e1(edge_flg) == i | e2(edge_flg) == i, 1);
        w = zeros(1, m);
        w(edge_flg) = randw(G2(edge_flg, edge_flg), es);
        edge_weights(i, :) = w;
    end
end
% node_weights = cell2mat(node_weights);
% edge_weights = cell2mat(edge_weights);
catch exception
    getReport(exception)
    keyboard;
end

end

function W = randw(G, s)
% Computes the weights of the nodes in the graph centering on node s.

% % Perturbes the transition matrix if it's singular (reducible graph).
% if rank(G) < size(G, 1)
%     while rank(G1) < size(G1, 1)
%         R = triu(rand(size(G)) * 0.001);
%         R = R + R';
%         G1 = R + G;
%     end
%     G = G1;
% end
G = G ./ repmat(sum(G, 2), 1, size(G, 2));

% Computes the stationary distribution. p_v = d(v) / 2|E|.
edge_count = sum(sum(triu(G)));
P = sum(G > 0, 2) / edge_count;
P = P';

% Computes the fundamental matrix.
eyes = zeros(size(G, 1), 1); eyes(s) = 1;
% btic; Z = (eye(size(G)) - G + repmat(P, size(G, 1), 1))\ eyes; toc;
% tic; Z = (eye(size(G)) - G + repmat(P, size(G, 1), 1)) \ eye(size(G)); toc;
Z = inv((eye(size(G)) - G + repmat(P, size(G, 1), 1))); 

% Expected number of visits to nodes in a roundtrip starting from s.
N = P / P(s);
N(s) = 2;

% Expected steps from s to other nodes.
D = (1 ./ P) .* (diag(Z)' - Z(s, :));
D(s) = 1 / P(s);

% Random weights 
W = N ./ D;
end