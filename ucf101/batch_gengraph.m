init_ucf101;

% =========================================================================
%
% Generates the graphs for each video.
%
% =========================================================================

W = cell(1, 3);
B = cell(1, 3);
for t = 1:3
    C = load([codebook_path filesep 'codebook_t' num2str(t) '.mat']);
    W{t} = C.W;
    B{t} = C.B;
end

parfor vid = 1 : length(video_list)
    vname = [video_list(vid).video_name];
    if ~exist([groups_path filesep vname '_groups.mat'], 'file')
        continue;
    end
   
    groups = load([groups_path filesep vname '_groups.mat']);
    
    G = gengraph_dtc(groups, graph_params);
    
    % Computes node's score over the codebook. 
    S = cell(1, 3);
    F = [groups.F_traj groups.F_hog groups.F_hof groups.F_mbh];
    F = F ./ repmat(sqrt(sum(F.^2, 2)), 1, size(F, 2));
    for t = 1:3
        S{t} = 1 ./ (1 + exp(-(F * W{t} + repmat(B{t}, size(F, 1), 1))));
    end
    
    save_graph([graph_path filesep num2str(vid) '_graph.mat'], G, S);
    
    [node_weights, edge_weights] = random_walk_weights_approx(G, ...
        randw_params.subgraph_radius);
    save_gw([graph_path filesep num2str(vid) '_gw.mat'], ...
        node_weights, edge_weights);
end