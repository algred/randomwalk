init_stanford40;
addpath(pathstring('Y:\tools\liblinear-1.92\matlab'));
addpath(genpath('/research/wvaction/tools/vlfeat-0.9.16-bin/vlfeat-0.9.16/toolbox/'));

% =========================================================================
%
% Trains and evaluates linear SVM on Bag-of-Subgraphs.
%
% =========================================================================
ncls = 40;

% Encodes BoW.
if ~exist([edgebox_path filesep 'bow.mat'], 'file')
    load([edgebox_path filesep 'edgebox_codebook.mat']);
    % X = zeros(length(annotation), size(C, 1));
    X = cell(length(annotation), 1);
    parfor i = id1 : id2 %1:length(annotation)
        [~, fname, ext] = fileparts(annotation{i}.imageName);
        if i == 2376 || i == 2377 
            continue;
        end
        edgebox = load([edgebox_path filesep fname '_edgebox.mat']);
        hog = edgebox.hog;
        x = vl_alldist2(hog', C', 'chi2');
        % x = sqrt(vl_alldist2(hog', C', 'l2'));
        % X(i, :) = exp(-min(x, [], 1));
        X{i} = min(x, [], 1);
    end
    save([edgebox_path filesep 'bow' num2str(id1) '.mat'], 'X');
    % save([edgebox_path filesep 'bow_l2.mat'], 'X');
else
    load([edgebox_path filesep 'bow.mat']);
    % load([edgebox_path filesep 'bow_l2.mat']);
end
% X(2376:2377, :) = [];
class_labels(2376:2377) = [];
used_for_training(2376:2377) = [];
X_train = exp(-X(used_for_training > 0, :));
X_test = exp(-X(used_for_training < 1, :));
L_train = class_labels(used_for_training > 0);
L_test = class_labels(used_for_training < 1);

% Trains linear SVM and evaluates on the test data.
c = 2.^[-5: 6];
test_acc = zeros(1, length(c));
AP = zeros(length(c), ncls);
multiclass_svm = 0;  
if ~multiclass_svm   
   for i = 1:length(c)
       wc = cell(1, ncls);
       parfor cid = 1 : ncls
           kp = 1e7 / (1e7 * sum(L_train == cid));
           kn = 1e7 / (1e7 * sum(L_train ~= cid));
           options = ['-s 3 -q -B 1 -c ' num2str(c(i)) ' -w1 ' ...
               num2str(kp, 10) ' -w-1 ' num2str(kn, 10) ];
           y = (L_train == cid) + (L_train ~= cid) * -1;
           model = train(y, sparse(X_train), options);
           wc{cid} = model.w' * model.Label(1);
       end
       wc = cell2mat(wc);
       D = [X_test ones(size(X_test, 1), 1)] * wc;
       for cid = 1:ncls
           y = (L_test == cid) + (L_test ~= cid) * -1;
           [recall, precision, info] = vl_pr(y, D(:, cid));
           AP(i, cid) = info.ap_interp_11;
       end
       test_acc(i) = mean(AP(i, :));
       fprintf('c = %f, test_acc = %f\n', c(i), test_acc(i));
   end
end

if multiclass_svm 
    wstr = [];
    for cid = 1:ncls
        wstr = [wstr ' -w' num2str(cid) ' ' ...
            num2str(1 / (sum(L_train == cid) * ncls))];
    end
    parfor i = 1:length(c)
        options = ['-s 4 -q -B 1 -c ' num2str(c(i)) wstr];
        model = train(L_train, sparse(X_train), options);
        [~, ~, d] = predict(L_test, sparse(X_test), model);
        AP = zeros(1, ncls);
        for cid = 1 : ncls
            y = (L_test == cid) + (L_test ~= cid) * -1;
            [recall, precision, info] = vl_pr(y, d(:, cid));
            AP(i, cid) = info.ap_interp_11;
        end
        test_acc(i) = mean(AP);
        fprintf('c = %f, test_acc = %f\n', c(i), test_acc(i));
    end
end

