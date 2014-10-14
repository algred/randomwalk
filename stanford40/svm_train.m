function [w b] = svm_train(X, neg_X, C)
num_neg = size(neg_X, 1); 
num_pos = size(X, 1);
kn = 1e7 / (2 * num_neg * 1e7);
kp = 1e7 / (2 * num_pos * 1e7);
svm_options = ['-q -B 1 -s 3 -c ' num2str(C) ...
    ' -w1 ' num2str(kp, 10) ' -w-1 ' num2str(kn, 10)];
y = [ones(num_pos, 1); ones(num_neg, 1) * -1];
model = train(y, sparse([X; neg_X]), svm_options);
w = model.w(1:end-1)';
b = model.w(end);

