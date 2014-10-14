%=============================================================================
%
% Learns action class specific object patterns.
%
%=============================================================================
init_stanford40;
addpath('/research/wvaction/code/apclustermex_linux64_2009/');
addpath(pathstring('Y:\tools\liblinear-1.92\matlab'));
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
if whiten_hog > 0
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
        D = abs(D) + mean(D) * 0.01;
        D = 1./D;
        D = diag(D);
        sigma = V * D;
        save('whiten_params.mat', 'sigma', 'mu');
    else
        load('whiten_params.mat');
    end
X = (X - repmat(mu, size(X, 1), 1)) * sigma;
end

% Normalizes the data.
% X = X ./ repmat(sqrt(sum(X .* X, 2)) + eps, 1, size(X, 2));

% Splits data to two folds for each class.
ncls = 40;
Xs = cell(ncls, 2);
negXs = cell(ncls, 2);
negLB = cell(ncls, 2);
for i = 1:ncls
    this_X = X(L(:, 1) == i & L(:, 2) > 0, :);
    N = size(this_X, 1);
    Xs{i, 1} = this_X(1 : floor(N / 2), :);
    Xs{i, 2} = this_X(floor(N / 2) + 1 : end, :);

    for j = 1:ncls
        if j == i
            continue;
        end
        this_negX = X(L(:, 1) == j & L(:, 2) > 0, :);
        this_negX = this_negX(randsample(...
            size(this_negX, 1), floor(size(this_negX, 1) / 10)), :);
        N = size(this_negX, 1);
        negXs{i, 1} = [negXs{i, 1}; this_negX(1 : floor(N / 2), :)];
        negXs{i, 2} = [negXs{i, 2}; this_negX(floor(N / 2) + 1 : end, :)];
        negLB{i, 1} = [negLB{i, 1}; ones(floor(N / 2), 1) * j];
        negLB{i, 2} = [negLB{i, 2}; ones(N - floor(N / 2), 1) * j];
    end
end
clear X;

% Clusters the data and trains a classifier for each cluster.
C = 2.^[-5:5];
thre = -0.5;
topK = 200;
for i = 1:ncls
    W = cell(1, 2);
    svm_c = cell(2, 1);
    entropy = cell(2, 1);
    for s = 1:2
        posX = Xs{i, s};
        negX = negXs{i, s};
        valX = [Xs{i, 2 - s + 1}; negXs{i, 2 - s + 1}];
        valn = size(valX, 1);
        valX = [valX ones(valn, 1)];
        valLB = [ones(size(Xs{i, 2-s+1}, 1), 1) * i; negLB{i, 2 - s + 1}]; 
        
        % Clustering by affinity propagation.
        A = exp(-squareform(pdist(posX)));
        lb = apclustermex(A, median(A(:)));
        K = length(unique(lb));
        map = [];
        map(unique(lb)) = 1:K;
        lb = map(lb);
        keyboard;
    
        % Trains classifier.
        entropy1 = ones(K, 1) * 1e5;
        svm_c1 = ones(K, 1) * 1e5;
        W1 = cell(1, K);
        for j = 1:K
            posX1 = posX(lb == j, :);
            negX1 = [negX; posX(lb ~= j, :)];
            np = size(posX1, 1);
            nn = size(negX1, 1);
            kp = 1e7 / (1e7 * np * 2);
            kn = 1e7 / (1e7 * nn * 2);
            y = [ones(np, 1); ones(nn, 1) * -1];
            options = ['-s 3 -q -B 1 -w1 ' ...
                    num2str(kp, 10) ' -w-1 ' num2str(kn, 10)];
            fprintf('cls = %d, split = %d, #cluster = %d, entropy ', i, s, j);
            W1{j} = zeros(size(posX1, 2), 1);
            for h = 1:length(C)
                options1 = [options ' -c ' num2str(C(h))];
                model = train(y, sparse([posX1; negX1]), options1);
                d = valX * model.w(:);
                [~, ix] = sort(d);
                p = histc(ix, 1:40);
                p = p / sum(p);
                this_entropy =  -1 * sum(p .* log(p + eps));
                fprintf(' %f ', this_entropy);
                if this_entropy < entropy1(j)
                    entropy1(j) = this_entropy;
                    svm_c1(j) = C(h);
                    W1{j} = model.w(:)';
                end
            end
            fprintf('\n');
        end
        W{s} = cell2mat(W1);
        svm_c{s} = svm_c1;
        entropy{s} = entropy1;
    end
    W = cell2mat(W);
    svm_c = cell2mat(svm_c);
    entropy = cell2mat(entropy);
    save([edgebox_path filesep 'obj_' num2str(i) '_whtn_' ...
        num2str(whiten_hog) '.mat'], 'W', 'svm_c', 'entropy'); 
end




