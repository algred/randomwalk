% init_high5;
% 
% % =========================================================================
% %
% % Selects edge codes with low appearance entropy across classes.
% %
% % =========================================================================
% ncls = 4;
% count = cell(1, ncls);
for vid = 267:length(video_list)
    if used_for_training(vid) < 1 
        continue;
    end
    label = class_labels(vid);
    graph = load([graph_path filesep num2str(vid) '_graph.mat']);
    graph_flip = load([graph_path filesep num2str(vid) '_graph_flip.mat']);
    for cid = 1:ncls
        c = edge_code_counts(graph.G, graph.RS{cid}, ...
            graph.PS{cid}, randw_params);
        if isempty(count{cid})
            count{cid} = zeros(ncls + 1, length(c));
        end
        count{cid}(label, :) = count{cid}(label, :) + c;
        count{cid}(label, :) = count{cid}(label, :) + ...
            edge_code_counts(graph_flip.G, graph_flip.RS{cid}, ...
            graph_flip.PS{cid}, randw_params);
    end
end

edge_idx = cell(1, ncls);
for cid = 1:ncls
    X = count{cid};
    P = X ./ repmat(sum(X, 1), size(X, 1), 1);
    p = -sum(P .* log(P), 1);
    [~, idx] = sort(p, 'ascend');
    edge_idx{cid} = idx(1:randw_params.edge_count_perclass);
end