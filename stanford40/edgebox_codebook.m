init_stanford40;
addpath(genpath('/research/wvaction/code/toolbox'));
addpath(genpath('/research/wvaction/code/vlfeat/vlfeat-0.9.18/'));

n = 10;
H = zeros(sum(used_for_training) * n, 8 * 8 * 31);
ind = 0;
for i = 1:length(annotation) 
    [~, fname, ext] = fileparts(annotation{i}.imageName);
    if i == 2376 || i == 2377 || used_for_training(i) < 1 
        continue;
    end
    load([edgebox_path filesep fname '_edgebox.mat']);
    idx = randsample(size(hog, 1), min(size(hog, 1), n));
    H(ind + 1 : ind + length(idx), :) = hog(idx, :);
    ind = ind + length(idx);
end
if ind < size(H, 1)
    H(ind + 1, :) = [];
end

N = 5e4;
K = 500;
idx = randsample(size(H, 1), min(size(H, 1), N));
[C, ~, ~] = vl_kmeans(H(idx, :)', K, 'numrepetitions', 10, 'algorithm', 'elkan');
C = C';
save([edgebox_path filesep 'edgebox_codebook.mat'], 'C');

