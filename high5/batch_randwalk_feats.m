init_high5;

% =========================================================================
%
% Generates random walk features of graphs.
%
% =========================================================================
randw_params.code_score_thresh = 0.2;
randw_params.pooling_mode = 2;
ncls = 4;

parfor vid = 1:length(video_list)
    if ~exist([randw_feat_path filesep '_' num2str(vid) '_rf.mat'], 'file')
        graph = load([graph_path filesep num2str(vid) '_graph.mat']);
        gw = load([graph_path filesep num2str(vid) '_gw.mat']);
        
        node_feats = cell(ncls, 1);
        edge_feats = cell(ncls, 1);
        for cid = 1:ncls
            tic; [node_feats{cid}, edge_feats{cid}] = ...
                randwalk_feats(graph.RS{cid}, graph.PS{cid}, graph.G, ...
                randw_params, gw.node_weights, gw.edge_weights); toc;
        end
        
        save_randwalk_feats([randw_feat_path filesep num2str(vid) ...
            '_rf.mat'], node_feats, edge_feats);
    end
    
    % Generates random walk features of fliped training videos.
    if used_for_training(vid) > 0 && ~exist([randw_feat_path filesep ...
            '_' num2str(vid) '_rf_flip.mat'], 'file')
        graph = load([graph_path filesep num2str(cid) '_'...
            num2str(vid) '_graph_flip.mat']);
        gw = load([graph_path filesep num2str(cid) '_' ...
            num2str(vid) '_gw_flip.mat']);
        
        node_feats = cell(ncls, 1);
        edge_feats = cell(ncls, 1);
        for cid = 1:ncls
            tic; [node_feats{cid}, edge_feats{cid}] = ...
                randwalk_feats(graph.RS{cid}, graph.PS{cid}, graph.G, ...
                randw_params, gw.node_weights, gw.edge_weights); toc;
        end
        
        save_randwalk_feats([randw_feat_path filesep num2str(vid) ...
            '_rf_flip.mat'], node_feats, edge_feats);
    end
end