init_high5;

% ==========================================================================
%
% Generates random walk features of graphs.
%
% ==========================================================================
train_idx = find(used_for_training > 0);
randw_params.subgraph_radius = 4;

parfor vid = 1:length(video_list)
    if ~exist([graph_path filesep num2str(vid) '_gw.mat'], 'file')
        graph = load([graph_path filesep num2str(vid) '_graph.mat']);
        
        % Computes random walk weights.
        [node_weights, edge_weights] = random_walk_weights_approx(...
            graph.G, randw_params.subgraph_radius);
        save_graphweights([graph_path filesep num2str(vid) '_gw.mat'], ...
            node_weights, edge_weights);
    end
    
    if ~exist([graph_path filesep num2str(vid) '_gw_flip.mat'], 'file') ...
            && exist([graph_path filesep num2str(vid) '_graph_flip.mat'], 'file')
        graph = load([graph_path filesep num2str(vid) '_graph_flip.mat']);
        
        % Computes random walk weights.
        [node_weights, edge_weights] = random_walk_weights_approx(...
            graph.G, randw_params.subgraph_radius);
        save_graphweights([graph_path filesep num2str(vid) '_gw_flip.mat'], ...
            node_weights, edge_weights);
    end
end