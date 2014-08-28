init_stanford40;

% =========================================================================
%
% Generates random walk features of graphs.
%
% =========================================================================
load(['data' filesep 'selected_edge_codes.mat']);
ncls = 40;
for id = 1:length(annotation)
    [~, imname, ext] = fileparts(annotation{id}.imageName);
    if exist([randw_feat_path filesep imname '_rf.mat'], 'file') || ...
            ~exist([graph_path filesep imname '_graph.mat'], 'file')
        continue;
    end
    graph = load([graph_path filesep imname '_graph.mat']);
    gw = load([graph_path filesep imname '_gw.mat']);
    
    tic;
    [node_feats, edge_feats] = randwalk_feats(double(graph.S), ...
        double(graph.L), graph.G, gw.node_weights, gw.edge_weights, ...
        edge_idx, randw_params);
    toc;
    save_randwalk_feats([randw_feat_path filesep imname '_rf.mat'], ...
        node_feats, edge_feats);
end
