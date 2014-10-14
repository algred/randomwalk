init_stanford40;
sfx = '_r3_CT';
for i = 1:length(annotation)
    [~, imname, ext] = fileparts(annotation{i}.imageName);
    if exist([graph_path filesep imname '_gw' sfx '.mat'], 'file') || ...
            i == 2376 || i == 2377
        continue;
    end
    
    graph = load([graph_path filesep imname '_graph.mat']);
    [node_weights, edge_weights] = random_walk_weights_approx2(...
        graph.G, graph_params.subgraph_radius);
    
    % Saves the generated graph.
    save_graphweights([graph_path filesep imname '_gw' sfx '.mat'], ...
        node_weights, edge_weights);
end
