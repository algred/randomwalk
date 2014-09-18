init_high5;
addpath(pathstring('Y:\tools\liblinear-1.92\matlab'));
addpath(genpath('/research/wvaction/tools/vlfeat-0.9.16-bin/vlfeat-0.9.16/toolbox/'));

% =========================================================================
%
% Trains linear SVM on Bag-of-Subgraph representation. 
%
% =========================================================================

% load(['data' filesep encoding_name '.mat']);
train_idx = find(used_for_training(info(:, 1)) > 0);
test_idx = find(used_for_training(info(:, 1)) < 1);

L_train = class_labels(info(train_idx, 1));
L_test = class_labels(info(test_idx, 1));

% Trains and evaluates.
X_train = F(train_idx, :);
X_test = F(test_idx, :); 
multiclass_svm = 0;

% nfold = 3;
% nc = zeros(4, 1);
% for c = 1:4
%     nc(c) = sum(c == L_train);
% end
% m = min(floor(nc/nfold));
% IDX = zeros(5, m, nfold);
% for c = 1:5
%     idx = find(c == L_train);
%     for f = 1:nfold
%         IDX(c, :, f) = idx((f-1)*m +1 : f*m);
%     end
% end

if ~multiclass_svm
   min_f = min(X_train); max_f = max(X_train);
   X_train = (X_train - repmat(min_f, size(X_train, 1), 1)) ./ ...
       repmat(max_f + eps, size(X_train, 1), 1);
   X_test= (X_test- repmat(min_f, size(X_test, 1), 1)) ./ ...
       repmat(max_f + eps, size(X_test, 1), 1);
   
   c = 2.^[-10:10];
   AP = zeros(length(c), 4);
   test_acc = zeros(length(c), 1);
   % train_acc = zeros(length(c), 1);
   for i = 1:length(c)
       %  train_acc(i) = nfoldCVBOWAP(X_train, L_train, 1:4, IDX, c(i));
       wc = zeros(size(X_train, 2) + 1, 4);
       for cid = 1 : 4
           kp = 1e7 / (1e7 * sum(L_train == cid));
           kn = 1e7 / (1e7 * sum(L_train ~= cid));
           options = ['-s 3 -q -B 1 -c ' num2str(c(i)) ...
               ' -w1 ' num2str(kp, 10) ' -w-1 ' num2str(kn, 10) ];
           y = (L_train == cid) + (L_train ~= cid) * -1;
           model = train(y, sparse(X_train), options);
           wc(:, cid) = model.w' * model.Label(1);
       end
       D = [X_test ones(size(X_test, 1), 1)] * wc;
       for cid = 1:4
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
    class_acc = zeros(length(c), 4);
    % train_acc = zeros(length(c), 4);
    test_acc = zeros(length(c), 4);
    AP = zeros(length(c), 4);
    
    min_f = min(X_train); max_f = max(X_train);
    X_train = (X_train - repmat(min_f, size(X_train, 1), 1)) ./ ...
        repmat(max_f + eps, size(X_train, 1), 1);
    X_test= (X_test- repmat(min_f, size(X_test, 1), 1)) ./ ...
        repmat(max_f + eps, size(X_test, 1), 1);
 
    wstr = [];
    for cid = 1:5
        wstr = [wstr ' -w' num2str(cid) ' ' ...
            num2str(1 / (sum(L_train == cid) * 5))];
    end
    
    for i = 1:length(c)
      %  train_acc(i) = nfoldCVBOWAP_m(X_train, L_train, 1:4, IDX, c(i));
        options = ['-s 4 -q -B 1 -c ' num2str(c(i)) wstr];
        model = train(L_train, sparse(X_train), options);
        [~, ~, d] = predict(L_test, sparse(X_test), model);
        for cid = 1 : 4
            y = (L_test == cid) + (L_test ~= cid) * -1;
            [recall, precision, info] = vl_pr(y, d(:, cid));
            AP(i, cid) = info.ap_interp_11;
        end
        test_acc(i) = mean(AP(i, :));
        fprintf('c = %f, test_acc = %f\n', c(i), test_acc(i));
    end
end
