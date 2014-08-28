init_ucf101;

% =========================================================================
%
% Selects edge codes with high frequency in positive class but low 
% appearance entropy across classes.
% 
% VERY SLOW!
%
% =========================================================================
% ccid = cell(1, 3);
% for t = 1:3
%     C = load([codebook_path filesep 'codebook_t' num2str(t) '.mat']);
%     ccid{t} = C.ccid;
% end
% 
% ncls = 101;
% count = cell(3, ncls);
% for t = 1:3
%     parfor cid = 1:ncls
%         for vid = 1:length(video_list)
%             if used_for_training(vid, t) < 1 || ~exist([graph_path ...
%                     filesep num2str(vid) '_graph.mat'], 'file')
%                 continue;
%             end
%             label = class_labels(vid);
%             graph = load([graph_path filesep num2str(vid) '_graph.mat']);
%             edge_count =sum(sum(graph.G > 0));
%             c = edge_code_counts(graph.G, ...
%                 graph.S{t}(:, ccid{t} == cid), randw_params);
%             if isempty(count{t, cid})
%                 count{t, cid} = zeros(ncls, length(c));
%             end
%             count{t, cid}(label, :) = count{t, cid}(label, :) + c;
%         end
%     end
% end
% save(['data' filesep 'edge_code_counts.mat'], 'count', '-v7.3');

edge_idx = cell(t, ncls);
for t = 1:3
    for cid = 1:ncls
        X = count{t, cid};
        
        % Selects edge codes that are frequent in positive class.
        [~, idx] = sort(X(cid, :), 'descend');
        idx1 = idx(1 : (randw_params.edge_count_perclass * 20));
        
        % Selects from the selected edge codes that have small entropy.
        P = X(:, idx1) ./ repmat(sum(X(:, idx1), 1), size(X, 1), 1);
        p = -sum(P .* log(P + eps), 1);
        [~, idx2] = sort(p, 'ascend');
        idx3 = idx1(idx2);
        edge_idx{t, cid} = idx3(1:randw_params.edge_count_perclass);
    end
end
save(['data' filesep 'selected_edge_codes.mat'], 'edge_idx');
