init_stanford40;

parfor i = 1:length(annotation)
    try
        [~, imname, ext] = fileparts(annotation{i}.imageName);
        if exist([graph_path filesep imname '_gw.mat'], 'file') || ...
                i == 2376 || i == 2377
            continue;
        end

        graph = load([graph_path filesep imname '_graph.mat']);
        [node_weights, edge_weights] = random_walk_weights_approx(...
            graph.G, graph_params.subgraph_radius);
        
        % Saves the generated graph.
        save_graphweights([graph_path filesep imname '_gw.mat'], ...
            node_weights, edge_weights);
    catch exception
        getReport(exception)
        fprintf('Error video id = %d\n', i);
        continue;
    end
end
