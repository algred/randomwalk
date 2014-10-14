init_stanford40;
addpath(genpath('/research/wvaction/code/toolbox'));
addpath('/home/grad2/shugaoma/lib/edgebox/release');
addpath(genpath('/research/wvaction/code/vlfeat/vlfeat-0.9.18/'));

opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e3;  % max number of boxes to detect

model=load('/home/grad2/shugaoma/lib/edgebox/release/models/forest/modelBsds');
model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

parfor i = 1:length(annotation) 
    [~, fname, ext] = fileparts(annotation{i}.imageName);
    if i == 2376 || i == 2377 ... 
            || exist([edgebox_path filesep fname '_edgebox.mat'], 'file')
        continue;
    end
    if exist([edgebox_path filesep fname '_edgebox.mat'], 'file')
        continue;
    end
    im = imread([img_path filesep annotation{i}.imageName]);
    bbs = edgeBoxes(im, model, opts);
    hog = zeros(size(bbs, 1), 8 * 8 * 31);
    for j = 1:size(bbs, 1)
        subim = imcrop(im, bbs(j, 1:4));
        subim = imresize(subim, [64, 64]);
        h = vl_hog(im2single(subim), 8);
        hog(j, :) = h(:)';
    end
    save_edgebox([edgebox_path filesep fname '_edgebox.mat'], bbs, hog);
end
