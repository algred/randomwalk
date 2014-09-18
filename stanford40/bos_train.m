init_stanford40;
addpath(pathstring('Y:\tools\liblinear-1.92\matlab'));
addpath(genpath('/research/wvaction/tools/vlfeat-0.9.16-bin/vlfeat-0.9.16/toolbox/'));

% =========================================================================
%
% Trains and evaluates linear SVM on Bag-of-Subgraphs.
%
% =========================================================================
ncls = 40;

% Loads data.
load(['data' filesep encoding_name]);
f = any(F, 2);
X_train = F(f & used_for_training > 0, :);
X_test = F(f & used_for_training < 1, :);
L_train = class_labels(f & used_for_training > 0);
L_test = class_labels(f & used_for_training < 1);

% Trains and evaluates linear SVM.
% multiclass_svm = 1;

if ~multiclass_svm
   min_f = min(X_train); max_f = max(X_train);
   X_train = (X_train - repmat(min_f, size(X_train, 1), 1)) ./ ...
       repmat(max_f + eps, size(X_train, 1), 1);
   X_test= (X_test- repmat(min_f, size(X_test, 1), 1)) ./ ...
       repmat(max_f + eps, size(X_test, 1), 1);
   
   %c = 2.^[-10:10];
   c = 2.^[11: 15];
   test_acc = zeros(length(c));
   AP = zeros(length(c), ncls);
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
    c = 2.^[-10:10];
    test_acc = zeros(1, length(c));
    
    min_f = min(X_train); max_f = max(X_train);
    X_train = (X_train - repmat(min_f, size(X_train, 1), 1)) ./ ...
        repmat(max_f + eps, size(X_train, 1), 1);
    
    X_test= (X_test- repmat(min_f, size(X_test, 1), 1)) ./ ...
        repmat(max_f + eps, size(X_test, 1), 1);
 
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
            AP(i) = info.ap_interp_11;
        end
        test_acc(i) = mean(AP);
        fprintf('c = %f, test_acc = %f\n', c(i), test_acc(i));
    end
end

