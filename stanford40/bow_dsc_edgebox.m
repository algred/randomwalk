init_stanford40;
addpath('/research/wvaction/code/dsc-release-v1');
addpath(pathstring('Y:\tools\liblinear-1.92\matlab'));
addpath(genpath('/research/wvaction/tools/vlfeat-0.9.16-bin/vlfeat-0.9.16/toolbox/'));
ncls = 40;

% Encodes each image.
n = 200;
if ~exist([edgebox_path filesep 'dsc_encoding_K120.mat'], 'file')
    W = [];
    B = [];
    for i = 1:ncls
        load([edgebox_path filesep 'dscluster_' num2str(i) '_K120.mat']);
        W = [W model.Ws];
        B = [B model.bs(:)'];
    end
    
    X = cell(length(annotation), 1);
    parfor i = 1:length(annotation)
        [~, fname, ext] = fileparts(annotation{i}.imageName);
        if ~exist([edgebox_path filesep fname '_edgebox_reduced.mat'])
            continue;
        end
        edgebox = load([edgebox_path filesep fname '_edgebox_reduced.mat']);
        hog = edgebox.hog(1:min(size(edgebox.hog, 1), n), :);
        hog = hog ./ repmat(sqrt(sum(hog .* hog, 2)) + eps, 1, size(hog, 2));
        X{i} = max(hog * W + repmat(B, size(hog, 1), 1), [], 1);
    end
    X = cell2mat(X);
    save([edgebox_path filesep 'dsc_encoding_K120.mat'], 'X', '-v7.3');
else
    load([edgebox_path filesep 'dsc_encoding_K120.mat']);
end
class_labels(2376:2377) = [];
used_for_training(2376:2377) = [];
X_train = X(used_for_training > 0, :);
X_test = X(used_for_training < 1, :);
% X_min = min(X_train);
% X_max = max(X_train);
% X_train = (X_train - repmat(X_min, size(X_train, 1), 1)) ./ ...
%     repmat(X_max - X_min + eps,  size(X_train, 1), 1);
% X_test = (X_test - repmat(X_min, size(X_test, 1), 1)) ./ ...
%     repmat(X_max - X_min + eps, size(X_test, 1), 1); 

L_train = class_labels(used_for_training > 0);
L_test = class_labels(used_for_training < 1);

% Trains linear SVM and evaluates on the test data.
c = 2.^[-5: 6];
ncls = 40;
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
