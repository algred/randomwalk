init_high5;

% =========================================================================
%
% Selects edge codes with high frequency in positive class but low 
% appearance entropy across classes.
%
% =========================================================================
% ncls = 4;
% count = cell(1, ncls);
% for vid = 1:length(video_list)
%     if used_for_training(vid) < 1 
%         continue;
%     end
%     label = class_labels(vid);
%     graph = load([graph_path filesep num2str(vid) '_graph.mat']);
%     graph_flip = load([graph_path filesep num2str(vid) '_graph_flip.mat']);
%     
%     edge_count =sum(sum(graph.G > 0));
%     edge_count_flip = sum(sum(graph_flip.G > 0));
%     
%     for cid = 1:ncls
%         c = edge_code_counts(graph.G, graph.RS{cid}, ...
%             graph.PS{cid}, randw_params);
%         if isempty(count{cid})
%             count{cid} = zeros(ncls + 1, length(c));
%         end
%         count{cid}(label, :) = count{cid}(label, :) + c;
%         count{cid}(label, :) = count{cid}(label, :) + ...
%             edge_code_counts(graph_flip.G, graph_flip.RS{cid}, ...
%             graph_flip.PS{cid}, randw_params);
%     end
% end

edge_idx = cell(1, ncls);
for cid = 1:ncls
    X = count{cid};

    % Selects edge codes that are frequent in positive class. 
    [~, idx] = sort(X(cid, :), 'descend');
    idx1 = idx(1 : (randw_params.edge_count_perclass * 5));
    
    % Selects from the selected edge codes that have small entropy.
    P = X(:, idx1) ./ repmat(sum(X(:, idx1), 1), size(X, 1), 1);
    p = -sum(P .* log(P + eps), 1);
    [~, idx2] = sort(p, 'ascend');
    idx3 = idx1(idx2);
    edge_idx{cid} = idx3(1:randw_params.edge_count_perclass);
end
save(['data' filesep 'selected_edge_codes.mat'], 'edge_idx');