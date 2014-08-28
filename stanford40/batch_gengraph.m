init_stanford40;
load(pathstring('Y:\data\image_data\Stanford40\MatlabAnnotations\annotation.mat'));

parfor i = 1:length(annotation)
    params = graph_params;
    [~, imname, ext] = fileparts(annotation{i}.imageName);
    if exist([graph_path filesep imname '_graph.mat'], 'file') || ...
            i == 2376 || i == 2377
        continue;
    end
    
    bbox = annotation{i}.bbox;
    bbox = [bbox(:, 1:2) bbox(:, 3:4) - bbox(:, 1:2) + 1];
    
    im = imread([img_path filesep annotation{i}.imageName]);
    img_width = size(im, 2); img_height = size(im, 1);
    params.overlap_thresh = min(img_width, img_height) * ...
        graph_params.overlap_thresh_ratio;
    
    % Loads poselet detections that are within bounding boxes.
    poselets = load([poselet150_path filesep imname '_poselets.mat']);
    poselets = poselets.poselet_hits;
    X = poselets.bounds';
    intxn_ratio = rectint(X, bbox) ./ ...
        repmat(X(:, 3) .* X(:, 4), 1, size(bbox, 1));
    flg = max(intxn_ratio, [], 2) >= bbox_overlap_ratio_thresh & ...
        poselets.score >= poselet_det_score_threh;
    X = X(flg, :);
    L = poselets.poselet_id(flg, :);
    S = poselets.score(flg, :);
    
    % Loads object detections.
    enlarged_box_x1 = max(floor(bbox(:, 1:2) - bbox(:, 3:4)), 1);
    enlarged_box_x2 = min(...
        repmat([img_width, img_height], size(bbox, 1), 1), ...
        enlarged_box_x1 + bbox(:, 3:4) * 3);
    enlarged_box = [enlarged_box_x1 enlarged_box_x2 - enlarged_box_x1];
    
    obj_dets = load([object_path filesep imname '_obj_rcnn.mat']);
    obj_dets = obj_dets.dets;
    det_idx = find(obj_cat_flg & (~cellfun(@isempty, obj_dets)));
    if ~isempty(det_idx)
        for j = 1:length(det_idx)
            id = det_idx(j);
            this_dets = obj_dets{id};
            this_dets(:, 1:4) = [this_dets(:, 1:2) ...
                this_dets(:, 3:4) - this_dets(:, 1:2) + 1];
            this_dets(:, 5) = 1 ./ (1 + exp(-this_dets(:, 5)));
            r = rectint(this_dets(:, 1:4), enlarged_box);
            flg1 = this_dets(:, 5) >= obj_det_score_thresh & ...
                max(r, [], 2) >= params.overlap_thresh_ratio;
            if any(flg1)
                X = [X; this_dets(flg1, 1:4)];
                S = [S; this_dets(flg1, 5)];
                L = [L; ones(sum(flg1), 1) * (NUM_POSELETS + obj_idx_map(id))];
            end
        end
    end
    
    % Generates the graph.
    G = gengraph_img(X, params);
    
    % Saves the generated graph.
    save_graph([graph_path filesep imname '_graph.mat'], G, X, S, L);
    
    % Generates random walk weights for the graph.
    [node_weights, edge_weights] = random_walk_weights_approx(...
        G, params.subgraph_radius);
    
    % Saves the generated graph.
    save_graphweights([graph_path filesep imname '_gw.mat'], ...
        node_weights, edge_weights);
end