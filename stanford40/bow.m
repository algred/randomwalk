init_stanford40;

% =========================================================================
%
% Encodes each image as bag-of-words.
%
% =========================================================================

F = zeros(length(annotation),  NUM_POSELETS + length(obj_idx));
for i = 1:length(annotation)
    [~, imname, ext] = fileparts(annotation{i}.imageName);
    
    if ~exist([graph_path filesep imname '_graph.mat'], 'file')
        continue;
    end

    load([graph_path filesep imname '_graph.mat']);
    f = accumarray(L, S, [], @max);
    F(i, 1:length(f)) = f;
end

save(['data' filesep 'bow.mat'], 'F');
