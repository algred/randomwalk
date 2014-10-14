init_stanford40;

% =========================================================================
%
% Generates random walk features of graphs.
%
% =========================================================================
% load(['data' filesep 'selected_edge_codes.mat']);
ncls = 40;
gw_sfx = '_r3_CT';
sfx = '_avgpool';
randw_params.pooling_mode = 0;
load([model_path filesep 'selected_edge_codes.mat']);
parfor id = 1:length(annotation)
    [~, imname, ext] = fileparts(annotation{id}.imageName);
    if exist([randw_feat_path filesep imname '_rf' sfx '.mat'], 'file') || ...
            ~exist([graph_path filesep imname '_graph.mat'], 'file')
        continue;
    end
    graph = load([graph_path filesep imname '_graph.mat']);
    gw = load([graph_path filesep imname '_gw' gw_sfx '.mat']);
    
    tic;
    [node_feats, edge_feats] = randwalk_feats(graph.S, graph.L, ...
        graph.G, gw.node_weights, gw.edge_weights, edge_idx, randw_params);
    toc;
    save_randwalk_feats([randw_feat_path filesep imname '_rf' sfx '.mat'], ...
        node_feats, edge_feats);
end
