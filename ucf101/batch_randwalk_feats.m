init_ucf101;

% =========================================================================
%
% Generates the graphs for each video.
%
% =========================================================================
ccid = cell(1, 3);
for t = 1:3
    C = load([codebook_path filesep 'codebook_t' num2str(t) '.mat']);
    ccid{t} = C.ccid;
end

ncls = 101;
load(['data' filesep 'selected_edge_codes.mat']);
parfor vid = vid1 : vid2
    vname = [video_list(vid).video_name];
    if ~exist([graph_path filesep num2str(vid) '_graph.mat'], 'file')
        continue;
    end
    
    if exist([randw_feat_path filesep num2str(vid) 'rf.mat'], 'file')
        continue;
    end
    
    graph = load([graph_path filesep num2str(vid) '_graph.mat']);
    gw = load([graph_path filesep num2str(vid) '_gw.mat']);
    
    node_feats = cell(3, ncls);
    edge_feats = cell(3, ncls);
    for cid = 1 : ncls
        for t = 1 : 3
        [node_feats{t, cid}, edge_feats{t, cid}] = ...
            randwalk_feats(graph.S{t}(:, ccid{t} == cid), ...
            graph.G, gw.node_weights, gw.edge_weights, ...
            edge_idx{t, cid}, randw_params);
        end
    end
    save_rf([randw_feat_path filesep num2str(vid) 'rf.mat'], ...
        node_feats, edge_feats);
end