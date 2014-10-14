%=============================================================================
%
% Learns action class specific object patterns.
%
%=============================================================================
init_stanford40;
addpath('/research/wvaction/code/dsc-release-v1');

% Loads edgebox data.
X = cell(length(annotation), 1);
L = cell(length(annotation), 1);
parfor i = 1:length(annotation)
    if used_for_training(i) < 1
        continue;
    end
    [~, fname, ext] = fileparts(annotation{i}.imageName);
    if ~exist([edgebox_path filesep fname '_edgebox_reduced.mat'])
        continue;
    end

    % Samples some object proposals.
    edgebox = load([edgebox_path filesep fname '_edgebox_reduced.mat']);
    X{i} = edgebox.hog;
    L{i} = ones(size(X{i}, 1), 1) * class_labels(i);
end
X = cell2mat(X);
L = cell2mat(L);

% Normalizes, splits and samples data for different classes.
ncls = 40;
X = X ./ repmat(sqrt(sum(X .* X, 2)) + eps, 1, size(X, 2));
Xs = cell(ncls, 1);
negXs = cell(ncls, 1);
N = 4000;
for i = 1:ncls
    Xs{i} = X(L(:) == i, :);
    sample_idx = randsample(size(Xs{i}, 1), ...
        min(size(Xs{i}, 1), N));
    Xs{i} = Xs{i}(sample_idx, :)';

    negXs{i} = X(L(:) ~= i, :);
    sample_idx = randsample(size(negXs{i}, 1), N * 4);
    negXs{i} = negXs{i}(sample_idx, :)';
end
clear X;

% Performs discriminative clustering within each class.
K = 120;
parfor i = id1 : id2
    % DSC clustering.
    C1 = 1e3; C2 = C1;
    [model, lb, obj_val, init_lb, obj_vals]  = M_McLinSVM.train_sgd(...
        Xs{i}, negXs{i}, K, C1, C2, [1, 100, 1]);

    % Saves.
    save_dscluster([edgebox_path filesep 'dscluster_' num2str(i) ...
        '_K120.mat'], model, lb, obj_val, init_lb, obj_vals);
end

