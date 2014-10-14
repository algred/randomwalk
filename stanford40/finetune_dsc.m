%====================================================================================
%
% Fine tunes the trained discriminative edgebox detector.
%
%====================================================================================
init_stanford40;
addpath(pathstring('Y:\tools\liblinear-1.92\matlab'));
ncls = 40;

% Encodes each image.
W = [];
B = [];
top_pos = 10;
top_neg = 3;
C = 1;
nitr = 100;
parfor i = cid1 : cid2
    model = load([edgebox_path filesep 'dscluster_' num2str(i) '_K100.mat']);
    model = model.model;
    K = size(model.Ws, 2);

    % Construct a fixed positive set.
    idx = find(class_labels == i & used_for_training > 0);
    ind = zeros(1, K);
    X = cell(K, 1);
    for k = 1:K
        X{k} = zeros(length(idx) * top_pos, 500);
    end
    for j = 1:length(idx)
        id = idx(j);
        [~, fname, ext] = fileparts(annotation{id}.imageName);
        if ~exist([edgebox_path filesep fname '_edgebox_reduced.mat'])
            continue;
        end
        edgebox = load([edgebox_path filesep fname '_edgebox_reduced.mat']);
        x = edgebox.hog * model.Ws + repmat(model.bs(:)', size(edgebox.hog, 1), 1);
        [x_sorted, ix] = sort(x, 1, 'descend');
        for k = 1:K
            ix1 = ix(1 : sum(x_sorted(1:top_pos, k) > 0), k);
            X{k}(ind(k) + 1 : ind(k) + length(ix1), :) = edgebox.hog(ix1, :);
            ind(k) = ind(k) + length(ix1);
        end
    end
    for k = 1:K
        if ind(k) < size(X{k}, 1)
            X{k}(ind(k) + 1 : end, :) = [];
        end
    end

    % Finetues the detectors by hard negative sample mining.
    for itr = 1 : nitr
        fprintf('cid = %d, itr = %d\n', i, itr);
        % Mines hard negatives.
        idx = find(class_labels ~= i & used_for_training > 0);
        idx = idx(randsample(length(idx), ceil(length(idx) / 5)));
        ind = zeros(1, K);
        X_neg = cell(K, 1);
        for k = 1:K
            X_neg{k} = zeros(length(idx) * top_neg, 500);
        end

        for j = 1:length(idx)
            id = idx(j);
            [~, fname, ext] = fileparts(annotation{id}.imageName);
            if ~exist([edgebox_path filesep fname '_edgebox_reduced.mat'])
                continue;
            end
            edgebox = load([edgebox_path filesep fname '_edgebox_reduced.mat']);
            x = edgebox.hog * model.Ws + ...
                repmat(model.bs(:)', size(edgebox.hog, 1), 1);
            [x_sorted, ix] = sort(x, 1, 'descend');
            for k = 1:K
                ix1 = ix(1 : sum(x_sorted(1:top_neg, k) > 0), k);
                X_neg{k}(ind(k) + 1 : ind(k) + length(ix1), :) = ...
                    edgebox.hog(ix1, :);
                ind(k) = ind(k) + length(ix1);
            end
        end
        for k = 1:K
            if ind(k) < size(X_neg{k}, 1)
                X_neg{k}(ind(k) + 1 : end, :) = [];
            end
        end

        % Retrains the detectors.
        for k = 1:K
            [w, b] = svm_train(X{k}, X_neg{k}, C);
            model.Ws(:, k) = w(:);
            model.bs(k) = b;
        end
    end
    save_finetuned_model([edgebox_path filesep 'dsc_finetuned_' ...
        num2str(i) '.mat'], model);
end
    
