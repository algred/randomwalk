%=============================================================================
%
% Learns action class specific object patterns.
%
%=============================================================================
init_stanford40;
addpath('/research/wvaction/code/dsc-release-v1');
addpath('/research/wvaction/code/apclustermex_linux64_2009/');
addpath(genpath('/research/wvaction/tools/vlfeat-0.9.16-bin/vlfeat-0.9.16/toolbox/'));

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

% Whitens the hog features.
if ~exist('whiten_params.mat', 'file')
    mu = mean(X, 1);
    if use_gpu
        X1 = gpuArray(X - repmat(mu, size(X, 1), 1));
        [~, D, V] = svd(X1);
        D = gather(D);
        V = gather(V);
        reset(gpudev);
    else
        [~, D, V] = svd(X - repmat(mu, size(X, 1), 1));
    end
    
    D = diag(D);
    D = abs(D);
    D(D <= 0) = min(D);
    D = 1./D;
    D = diag(D);
    sigma = V * D;
    save('whiten_params.mat', 'sigma', 'mu');
else
    load('whiten_params.mat');
end
X = (X - repmat(mu, size(X, 1), 1)) * sigma;

% Normalizes, splits and samples data for different classes.
ncls = 40;
X = X ./ repmat(sum(X .* X, 2) + eps, 1, size(X, 2));
Xs = cell(ncls, 1);
negXs = cell(ncls, 1);
N = 2000;
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
parfor i = 1:ncls
%     % Initial clustering by affinity propagation.
%     A = exp(-squareform(pdist(Xs{i}')));
%     init_lb = apclustermex(A, median(A(:)));
%     K = length(unique(init_lb));
%     map = [];
%     map(unique(init_lb)) = 1:K;
%     init_lb = map(init_lb);

    % DSC clustering.
    C1 = 1e3; C2 = C1;
    [model, lb, obj_val, init_lb, obj_vals]  = M_McLinSVM.train_sgd(...
        Xs{i}, negXs{i}, K, C1, C2, [1, 100, 1], init_lb(:));

    % Saves.
    save_dscluster([edgebox_path filesep 'dscluster_' num2str(i) '_whtn_' ...
        num2str(whiten_hog) '.mat'], model, lb, obj_val, init_lb, obj_vals);
end

