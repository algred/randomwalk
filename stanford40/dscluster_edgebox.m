init_stanford40;
addpath('/research/wvaction/code/dsc-release-v1');
addpath('/research/wvaction/code/apclustermex_linux64_2009/');
addpath(genpath('/research/wvaction/tools/vlfeat-0.9.16-bin/vlfeat-0.9.16/toolbox/'));

% Loads edgebox data and generates negative samples.
n = 5;
boxsz = 100;
X = cell(length(annotation), 1);
negX = cell(length(annotation), 1);
parfor i = 1:length(annotation)
    [~, fname, ext] = fileparts(annotation{i}.imageName);
    if ~exist([edgebox_path filesep fname '_edgebox.mat']) ...
            || used_for_training(i) < 1
        continue;
    end

    % Samples some object proposals.
    edgebox = load([edgebox_path filesep fname '_edgebox.mat']);
    idx = randsample(size(edgebox.hog, 1), min(n, size(edgebox.hog, 1)));
    X{i} = edgebox.hog(idx, :);

    % Samples some non-object windows.
    im = imread([img_path filesep annotation{i}.imageName]);
    rows = size(im, 1); cols = size(im, 2);
    n1 = n * 100;
    C = max(1, floor(rand(n1, 2) .* ...
        repmat([cols - boxsz rows - boxsz], n1, 1)));
    x = C(:, 1); y = C(:, 2);
    W = max(1, floor(randn(n1, 2) * boxsz));
    w = min(cols - x, W(:, 1));
    h = min(rows - y, W(:, 2));
    ovlp = rectint([x y w h], edgebox.bbs);
    [ovlp_sorted, ix] = sort(max(ovlp, [], 2), 'ascend');
    idx1 = ix(ovlp_sorted < 0.2); 
    if isempty(idx1)
        continue;
    end
    idx1 = idx1(1 : min(length(idx1), n));
    bbs1 = [x(idx1) y(idx1) w(idx1) h(idx1)];
    hog1 = zeros(length(idx1), size(edgebox.hog, 2));
    for j = 1:length(idx1)
        subim = imcrop(im, bbs1(j, :));
        subim = imresize(subim, [64, 64]);
        h = vl_hog(im2single(subim), 8);
        hog1(j, :) = h(:)';
    end
    negX{i} = hog1;
end
X = cell2mat(X); negX = cell2mat(negX);

% Performs DSC clustering.
X1 = M_Utils.normalize_L2(X'); 
negX1 = M_Utils.normalize_L2(negX');

A = exp(-vl_alldist2(X1, X1, 'chi2'));
init_lb = apclustermex(A, median(A(:)));
K = length(unique(init_lb));
map(unique(init_lb)) = 1:K;
init_lb = map(init_lb);

C1 = 1e3; C2 = C1;
[model, lb, obj_val, init_lb, obj_vals]  = M_McLinSVM.train_sgd(...
    X1, negX1, K, C1, C2, [1, 150, 5], init_lb(:));
save([edgebox_path filesep 'dsc_clusters.mat'], 'model', 'lb', ...
   'obj_val', 'init_lb', 'obj_vals', 'X', 'negX');




