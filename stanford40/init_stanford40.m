% Paths to data.
img_path = pathstring('Y:\data\image_data\Stanford40\JPEGImages');
img_format = 'jpg'; 
output_root = pathstring('Y:\data\image_data\Stanford40\static_action_output');
poselet_path = [output_root filesep 'poselets'];
poselet150_path = [output_root filesep 'poselet150'];
object_path = [output_root filesep 'objects_with_part'];
graph_path = [output_root filesep 'graph'];
randw_feat_path = [output_root filesep 'randw_feats'];

% Selected object categories from pre-trained R-CNN object detectors.
obj_idx = [8, 22, 23, 24, 27, 29, 30, 37, 38, 42, 43, 47, 48, 54, 57, ...
    58, 68, 73, 79, 89, 92, 98, 101, 109, 110, 128, 131, 145, 164, 165, ...
    169, 173, 177, 189, 190, 191, 192, 196, 199, 162];
obj_cat_flg = zeros(200, 1);
obj_cat_flg(obj_idx) = 1;
obj_cat_flg = obj_cat_flg > 0;
obj_idx_map(obj_idx) = 1:length(obj_idx);

% Paths to code.
addpath(pathstring('Y:\backed_up\randomwalk'));
addpath(pathstring('Y:\code\poselets_matlab_april2013\detector\poselet_detection'));

% Loads annotations and splits.
load(pathstring('Y:\data\image_data\Stanford40\stanford40_data.mat'));

% Parameters.
graph_params.OVERLAP_OVERLAP = 1;
graph_params.OVERLAP_ABOVE = 2;
graph_params.APART_OVERLAP = 3;
graph_params.APART_ABOVE = 4;
graph_params.overlap_thresh_ratio = 0.05;
graph_params.neighbor_K = 6;
graph_params.subgraph_radius = 4;

NUM_POSELETS = 150;
bbox_overlap_ratio_thresh = 0.7;
obj_det_score_thresh = 0.35;
poselet_det_score_threh = 0.05;

randw_params.pooling_mode = 2;
randw_params.code_score_thresh = 0.2;
randw_params.edge_count_perclass = 200;
randw_params.code_count = NUM_POSELETS + length(obj_idx);
randw_params.edge_types = [graph_params.OVERLAP_OVERLAP, ...
    graph_params.OVERLAP_ABOVE, graph_params.APART_OVERLAP, ...
    graph_params.APART_ABOVE];

