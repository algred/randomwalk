init_stanford40;
addpath(pathstring('Y:\tools\rcnn\rcnn'));
addpath(pathstring('Y:\tools\caffe-openhero\matlab\caffe'));

% Loads and initializes CNN model.
model_def_file = pathstring(...
    'Y:\tools\caffe-openhero\examples\imagenet\imagenet_fc7_feature.prototxt');
model_file = pathstring(...
    'Y:\tools\caffe-openhero\examples\imagenet\caffe_reference_imagenet_model');
caffe('init', model_def_file, model_file);
caffe('set_mode_gpu');
caffe('set_phase_test');
layer = '_fc7_';

% Initializes the cnn feature extraction parameters.
load(pathstring('Y:\tools\caffe-openhero\matlab\caffe\ilsvrc_2012_mean'));
cnn_params.image_mean = image_mean;
cnn_params.batch_size = 10;
cnn_params.width = 227;
cnn_params.height = 227;

% Extracts CNN feature from top n edgebox.
n = 200;
for i = 1:length(annotation) 
    [~, fname, ext] = fileparts(annotation{i}.imageName);
    if ~exist([edgebox_path filesep fname '_edgebox.mat'], 'file')
        continue;
    end
    if exist([edgebox_path filesep fname '_edgebox_cnn.mat'], 'file')
        continue;
    end
    edgebox = load([edgebox_path filesep fname '_edgebox.mat']);
    n1 = size(edgebox.bbs, 1);
    bbs = edgebox.bbs(1:min(n1, n), :);
    im = imread([img_path filesep annotation{i}.imageName]);
    if ndims(im) < 3
        im = repmat(im, [1, 1, 3]);
    end
    subim = cell(size(bbs, 1), 1);
    for j = 1:size(bbs, 1)
        subim{j} = imcrop(im, bbs(j, 1:4));
    end
    cnn = extract_cnnfeats(subim, cnn_params);
    save([edgebox_path filesep fname '_edgebox_cnn.mat'], 'cnn');
end
