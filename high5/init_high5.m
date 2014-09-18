%==========================================================================
%
% Paths.
%
%==========================================================================
if ~ispc; addpath('/home/grad2/shugaoma/local/mexlib'); end;
addpath(pathstring('Y:\backed_up\randomwalk'));

stsegment_path = pathstring('Y:\projects\actionlet\v2\data\high5\actionlet');
examplar_path = pathstring('/research/wvaction/data/video_data/high5/exemplars');
groups_path = pathstring('/research/wvaction/data/video_data/high5/groups');
graph_path = pathstring('/research/wvaction/data/video_data/high5/randomwalk/graph');
randw_feat_path = pathstring('/research/wvaction/data/video_data/high5/randomwalk/feat');
model_path = pathstring('/research/wvaction/backed_up/composities/high5/data');

%==========================================================================
%
% Params.
%
%==========================================================================

graph_params.BEFORE = 1;
graph_params.OVERLAP = 2;
graph_params.ABOVE = 3;
graph_params.DUPLICATE = 4;
graph_params.FAKE = 5;
graph_params.BEFORE_OVERLAP = 6;
graph_params.BEFORE_ABOVE = 7;
graph_params.BEFORE_BELOW = 8;
graph_params.sep = 1e4;

graph_params.tovlp_thre = 3;
graph_params.sovlp_thre = 0.6;
graph_params.tbefore_thre = 3;
graph_params.sabove_thre = 20;
graph_params.redundant_thre = 0.75;
graph_params.neighbor_K = 6;

randw_params.edge_types = [graph_params.BEFORE, graph_params.OVERLAP, ...
    graph_params.ABOVE, graph_params.BEFORE_OVERLAP, ...
    graph_params.BEFORE_ABOVE, graph_params.BEFORE_BELOW];
randw_params.code_score_thresh = 0.2;
randw_params.edge_count_perclass = 1000;
%==========================================================================
%
% Data.
%
%==========================================================================

load(pathstring('Y:\projects\actionlet\v2\data\high5\high5_data.mat'));
