init_stanford40;
addpath('/research/wvaction/code/dsc-release-v1');
addpath(pathstring('Y:\tools\liblinear-1.92\matlab'));
addpath(genpath('/research/wvaction/tools/vlfeat-0.9.16-bin/vlfeat-0.9.16/toolbox/'));
class_labels(2376:2377) = [];
used_for_training(2376:2377) = [];
ncls = 40;

% Computes activation entropy for each dsc.
% K = 120;
% thre = 0;
% load([edgebox_path filesep 'dsc_encoding_K120.mat']);
% X_train = X(used_for_training > 0, :);
% L_train = class_labels(used_for_training > 0);
% F = X_train > 0;
% dsc_entropy = zeros(1, size(X_train, 2));
% dsc_entropy_rank = zeros(1, size(X_train, 2));
% for cid = 1:ncls
%     idx = (cid - 1) * K + 1 : cid * K;
%     F1 = F(:, idx);
%     dsc_entropy1 = zeros(size(idx));
%     for j = 1:length(idx)
%         p = accumarray(L_train(:), F1(:, j), [], @sum);
%         p = p ./ sum(p);
%         dsc_entropy1(j) = sum(-1 * p .* log(p + eps));
%     end
%     dsc_entropy(idx) = dsc_entropy1;
%     [~, ix] = sort(dsc_entropy1);
%     dsc_entropy_rank(ix) = 1:length(idx);
% end
% save([edgebox_path filesep 'dsc_entropy_rank_K120.mat'], ...
%     'dsc_entropy_rank', 'dsc_entropy');

load([edgebox_path filesep 'dsc_entropy_rank_K120.mat'], ...
    'dsc_entropy_rank', 'dsc_entropy');
load([edgebox_path filesep 'dsc_encoding_K120.mat']);
X_train = X(used_for_training > 0, :);
L_train = class_labels(used_for_training > 0);

c = 2.^[-1: 5];
RR = 20:5:120;
test_acc = cell(length(RR), 1);
parfor r = 1:length(RR)
    R = RR(r);
    % Trains linear SVM and evaluates on the test data.
    X_test = X(used_for_training < 1, dsc_entropy_rank < R);
    L_test = class_labels(used_for_training < 1);
    X_train1 = X_train(:, dsc_entropy_rank < R);
    ncls = 40;
    AP = zeros(length(c), ncls);
    multiclass_svm = 0;  
   for i = 1:length(c)
       wc = cell(1, ncls);
       for cid = 1 : ncls
           kp = 1e7 / (1e7 * sum(L_train == cid));
           kn = 1e7 / (1e7 * sum(L_train ~= cid));
           options = ['-s 3 -q -B 1 -c ' num2str(c(i)) ' -w1 ' ...
               num2str(kp, 10) ' -w-1 ' num2str(kn, 10) ];
           y = (L_train == cid) + (L_train ~= cid) * -1;
           model = train(y, sparse(X_train1), options);
           wc{cid} = model.w' * model.Label(1);
       end
       wc = cell2mat(wc);
       D = [X_test ones(size(X_test, 1), 1)] * wc;
       for cid = 1:ncls
           y = (L_test == cid) + (L_test ~= cid) * -1;
           [recall, precision, info] = vl_pr(y, D(:, cid));
           AP(i, cid) = info.ap_interp_11;
       end
       test_acc{r}(i) = mean(AP(i, :));
       fprintf('R = %d, c = %f, test_acc = %f\n', R, c(i), test_acc{r}(i));
   end
end
