%====================================================================================
%
% Reduces the number of edgeboxs and reduce the feature dimension by PCA.
%
% This is only done for the training data. For testing data this is done on the fly.
%====================================================================================

init_stanford40;

n = 200;
load([edgebox_path filesep 'pca_params.mat']);
parfor i = 1:length(annotation)
    if used_for_training(i) > 0
        continue;
    end
    [~, fname, ext] = fileparts(annotation{i}.imageName);
    if ~exist([edgebox_path filesep fname '_edgebox.mat'])
        continue;
    end
    edgebox = load([edgebox_path filesep fname '_edgebox.mat']);
    hog = edgebox.hog(1:min(size(edgebox.hog, 1), n), :);
    hog = bsxfun(@minus, hog, mu) * C(:, 1:500);
    save_hog([edgebox_path filesep fname '_edgebox_reduced.mat'], hog);
end

