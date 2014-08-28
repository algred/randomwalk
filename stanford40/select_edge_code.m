init_stanford40;

% =========================================================================
%
% Selects edge codes with high frequency in positive class but low 
% appearance entropy across classes.
%
% =========================================================================
ncls = 40;
edge_code_count = (randw_params.code_count^2) * length(randw_params.edge_types);
count = zeros(ncls, edge_code_count);
for id = 1:length(annotation)
    if used_for_training(id) < 1 
        continue;
    end
    [~, imname, ext] = fileparts(annotation{id}.imageName);
    if ~exist([graph_path filesep imname '_graph.mat'], 'file')
        continue;
    end

    graph = load([graph_path filesep imname '_graph.mat']);
    edge_count =sum(sum(graph.G > 0));
    label = class_labels(id);
    
    c = edge_code_counts(graph.G, graph.L, randw_params);
    count(label, :) = count(label, :) + c;
end
save(['data' filesep 'edge_code_count.mat'], 'count');

edge_idx = cell(1, ncls);
for cid = 1:ncls
    % Selects edge codes that are frequent in positive class. 
    [~, idx] = sort(count(cid, :), 'descend');
    idx1 = idx(1 : (randw_params.edge_count_perclass * 10));
    
    % Selects from the selected edge codes that have small entropy.
    P = count(:, idx1) ./ repmat(sum(count(:, idx1), 1), ncls, 1);
    p = -sum(P .* log(P + eps), 1);
    [~, idx2] = sort(p, 'ascend');
    idx3 = idx1(idx2);
    edge_idx{cid} = idx3(1:randw_params.edge_count_perclass);
end

edge_idx2 = cell(1, ncls);
for cid = 1:ncls
    % Selects edge codes that are frequent in positive class. 
    [~, idx] = sort(count(cid, :), 'descend');
    idx1 = idx(1 : (randw_params.edge_count_perclass * 30));
    
    % Selects from the selected edge codes that have small entropy.
    P = count(:, idx1) ./ repmat(sum(count(:, idx1), 1), ncls, 1);
    p = -sum(P .* log(P + eps), 1);
    [~, idx2] = sort(p, 'ascend');
    idx3 = idx1(idx2);
    edge_idx2{cid} = idx3(1:randw_params.edge_count_perclass);
end

edge_idx = unique([cell2mat(edge_idx) cell2mat(edge_idx2)]);
save(['data' filesep 'selected_edge_codes.mat'], 'edge_idx');
