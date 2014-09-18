init_stanford40;
addpath(genpath('/research/wvaction/code/vlfeat/vlfeat-0.9.18/'));
for i = 2378:length(annotation)
    [~, fname, ext] = fileparts(annotation{i}.imageName);
    if exist([sift_path filesep fname '_sift.mat'], 'file')
        continue;
    end
    im = imread([img_path filesep annotation{i}.imageName]);
    im = single(rgb2gray(im));
    [f, d] = vl_sift(im);
    save([sift_path filesep fname '_sift.mat'], 'f', 'd');
end
