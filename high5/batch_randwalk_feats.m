init_high5;

% ==========================================================================
%
% Generates random walk features of graphs.
%
% ==========================================================================
train_idx = find(used_for_training > 0);
radius = 2;
randw_params.subgraph_radius = radius;
randw_params.code_score_thresh = 0.2;
randw_params.pooling_mode = 2;
% dbstop if error;

for cid = 1:4
    for vid = 2:12 %length(video_list)
        graph = load([graph_path filesep num2str(cid) '_' num2str(vid) '_graph.mat']);
        G = graph.G .* (graph.GS ~= 0);
        % No duplicate or fake edges.
        G1 = (G >= 6);
        G(G1(:)) = G(G1(:)) - 2; 
        
        tic; [node_weights, edge_weights] = random_walk_weights(...
            G, randw_params.subgraph_radius); toc;
        
        tic; [node_feats, edge_feats] = ...
            randwalk_feats(graph.RS, graph.PS, G, ...
            randw_params, node_weights, edge_weights); toc;
        
        save_randwalk_feats([randw_feat_path filesep num2str(cid) ... 
            '_' num2str(vid) '_' num2str(radius) '_rf.mat'], ...
            node_feats, edge_feats, node_weights, edge_weights);
        
        % Generates random walk features of fliped training videos.
        if used_for_training(vid) > 0
            graph = load([graph_path filesep num2str(cid) ...
                '_' num2str(vid) '_graph_flip.mat']);
            G = graph.G .* (graph.GS ~= 0);
            G1 = (G >= 6);
            G(G1(:)) = G(G1(:)) - 2;
            [node_feats, edge_feats, node_weights, edge_weights] = ...
                randwalk_feats(graph.RS, graph.PS, G,  randw_params);
            save_randwalk_feats([randw_feat_path filesep num2str(cid) ... 
                '_' num2str(vid) '_' num2str(radius) '_rf_flip.mat'], ...
                node_feats, edge_feats, node_weights, edge_weights);
        end
    end
end