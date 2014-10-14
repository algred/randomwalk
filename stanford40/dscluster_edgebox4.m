%=============================================================================
%
% Learns action class specific object patterns.
%
%=============================================================================
init_stanford40;
addpath('/research/wvaction/code/dsc-release-v1');

% Loads edgebox data.
if ~exist([edgebox_path filesep 'data.mat'], 'file')
    n = 50;
    X = cell(length(annotation), 1);
    L = cell(length(annotation), 1);
    parfor i = 1:length(annotation)
        [~, fname, ext] = fileparts(annotation{i}.imageName);
        if ~exist([edgebox_path filesep fname '_edgebox.mat']) ...
                || used_for_training(i) < 1
            continue;
        end
    
        % Samples some object proposals.
        edgebox = load([edgebox_path filesep fname '_edgebox.mat']);
        X{i} = edgebox.hog(1:min(size(edgebox.hog, 1), n), :);
        L{i} = repmat([class_labels(i) used_for_training(i)], size(X{i}, 1), 1);
    end
    X = cell2mat(X);
    L = cell2mat(L);
    save([edgebox_path filesep 'data.mat'], 'X', 'L', '-v7.3');
else
    load([edgebox_path filesep 'data.mat']);
end

% Reduces feature dimension by PCA.
if ~exist([edgebox_path filesep 'pca_params.mat'], 'file')
    X1 = X(randsample(size(X, 1), 5000), :);
    mu = mean(X1);
    [C, S, lat] = pca(X1);
    save([edgebox_path filesep 'pca_params.mat'], 'mu', 'C', 'lat');
else 
    load([edgebox_path filesep 'pca_params.mat']);
end
X = bsxfun(@minus, X, mu) * C(:, 1:500);

% Normalizes, splits and samples data for different classes.
ncls = 40;
X = X ./ repmat(sqrt(sum(X .* X, 2)) + eps, 1, size(X, 2));
Xs = cell(ncls, 1);
negXs = cell(ncls, 1);
N = 1500;
for i = 1:ncls
    Xs{i} = X(L(:, 1) == i & L(:, 2) > 0, :);
    sample_idx = randsample(size(Xs{i}, 1), ...
        min(size(Xs{i}, 1), N));
    Xs{i} = Xs{i}(sample_idx, :)';

    negXs{i} = X(L(:, 1) ~= i & L(:, 2) > 0, :);
    sample_idx = randsample(size(negXs{i}, 1), N * 4);
    negXs{i} = negXs{i}(sample_idx, :)';
end
clear X;

% Performs discriminative clustering within each class.
K = 50;
parfor i = id1 : id2
    % DSC clustering.
    C1 = 1e3; C2 = C1;
    [model, lb, obj_val, init_lb, obj_vals]  = M_McLinSVM.train_sgd(...
        Xs{i}, negXs{i}, K, C1, C2, [1, 100, 1]);

    % Saves.
    save_dscluster([edgebox_path filesep 'dscluster_' num2str(i) ...
        '_K100.mat'], model, lb, obj_val, init_lb, obj_vals);
end

