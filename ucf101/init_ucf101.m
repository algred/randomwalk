%% Paths to ucf101 data and output.
dtpath = pathstring('/research/action_videos/video_data/thumos14/dt');
dtcode_path = pathstring('/research/action_videos/video_data/thumos14/THUMOS14_IDTF_Codebook');
groups_path = pathstring('/research/action_videos/video_data/thumos14/output/groups');
codebook_path = pathstring('X:\video_data\thumos14\output\codebook');
output_root = pathstring('/research/action_videos/video_data/thumos14/output/randomwalk');
graph_path = [output_root filesep 'graph'];
randw_feat_path = [output_root filesep 'feat'];

%% Paths to code.
addpath(pathstring('Y:\backed_up\randomwalk'));

%% THUMOS13 data: train / test splits and more.
load(pathstring('y:\backed_up\thumos\thumos13\thumos13_data.mat'));

%% Parameters.
graph_params.BEFORE_OVERLAP = 1;
graph_params.BEFORE_ABOVE = 2;
graph_params.BEFORE_BELOW = 3;
graph_params.OVERLAP_OVERLAP = 4;
graph_params.OVERLAP_ABOVE = 5;
graph_params.time_diff_thresh = 10;
graph_params.time_overlap_thresh = 5;
graph_params.space_overlap_thresh = 15;
graph_params.K = 5;

randw_params.subgraph_radius = 4;
randw_params.code_score_thresh = 1 ./ (1 + exp(1));
randw_params.pooling_mode = 2;
randw_params.edge_types = [graph_params.BEFORE_OVERLAP, ...
    graph_params.BEFORE_ABOVE, graph_params.BEFORE_BELOW, ...
    graph_params.OVERLAP_OVERLAP, graph_params.OVERLAP_ABOVE];
randw_params.edge_count_perclass = 100;


