init_high5;

% =========================================================================
%
% Generates random walk features of graphs.
%
% =========================================================================
randw_params.code_score_thresh = 0.2;
randw_params.pooling_mode = 1;
ncls = 4;
load(['data' filesep 'selected_edge_codes.mat']);
parfor vid = 1:length(video_list)
    if ~exist([randw_feat_path filesep num2str(vid) '_rf_maxpool.mat'], 'file')
        graph = load([graph_path filesep num2str(vid) '_graph.mat']);
        gw = load([graph_path filesep num2str(vid) '_gw.mat']);
        
        node_feats = cell(ncls, 1);
        edge_feats = cell(ncls, 1);
        for cid = 1:ncls
            tic;
            [node_feats{cid}, edge_feats{cid}] = ...
                randwalk_feats(graph.RS{cid}, graph.PS{cid}, graph.G, ...
                gw.node_weights, gw.edge_weights, ...
                edge_idx{cid}, randw_params);
            toc;
        end
        
        save_randwalk_feats([randw_feat_path filesep num2str(vid) ...
            '_rf_maxpool.mat'], node_feats, edge_feats);
    end
    
    % Generates random walk features of fliped training videos.
    if used_for_training(vid) > 0 && ~exist([randw_feat_path filesep ...
            num2str(vid) '_rf_maxpool_flip.mat'], 'file')
        graph = load([graph_path filesep num2str(vid) '_graph_flip.mat']);
        gw = load([graph_path filesep num2str(vid) '_gw_flip.mat']);
        
        node_feats = cell(ncls, 1);
        edge_feats = cell(ncls, 1);
        for cid = 1:ncls
            tic;
            [node_feats{cid}, edge_feats{cid}] = ...
                randwalk_feats(graph.RS{cid}, graph.PS{cid}, graph.G, ...
                gw.node_weights, gw.edge_weights, ...
                edge_idx{cid}, randw_params);
            toc;
        end
        
        save_randwalk_feats([randw_feat_path filesep num2str(vid) ...
            '_rf_maxpool_flip.mat'], node_feats, edge_feats);
    end
end
