init_stanford40;
addpath(genpath('/research/wvaction/code/vlfeat/vlfeat-0.9.18/'));
vl_setup;

n = 10;
H = zeros(sum(used_for_training) * n, 128);
ind = 0;
for i = 1:length(annotation)
    [~, fname, ext] = fileparts(annotation{i}.imageName);
    if i == 2376 || i == 2377 || used_for_training(i) < 1
        continue;
    end
    load([sift_path filesep fname '_sift.mat']);
    idx = randsample(size(d, 2), min(size(d, 2), n));
    H(ind + 1 : ind + length(idx), :) = d(:, idx)';
    ind = ind + length(idx);
end
if ind < size(H, 1)
    H(ind + 1, :) = [];
end
H = double(H);
H = H ./ repmat(sum(H, 2) + eps, 1, size(H, 2));
N = 5e4;
K = 800;
idx = randsample(size(H, 1), min(size(H, 1), N));
[C, ~, ~] = vl_kmeans(H(idx, :)', K, 'numrepetitions', 10, 'algorithm', 'elkan');
C = C';
save([sift_path filesep 'sift_codebook.mat'], 'C');
