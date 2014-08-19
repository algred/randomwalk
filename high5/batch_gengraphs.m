init_high5;

% ==========================================================================
%
% Generates graphs.
%
% ==========================================================================
train_idx = find(used_for_training > 0);
dsc_scales = load([model_path filesep 'dsc_scale.mat']);
load([model_path filesep 'cluster_scales.mat']);
for cid = 1:4
    offr = dsc_scales.offr; offp = dsc_scales.offp;
    % Loads the clusters.
    clusters = load([model_path filesep 'cluster_' num2str(cid) '.mat']);
    root_model = clusters.root_dscluster.model;
    part_model = clusters.part_dscluster.model;
    scale = dsc_scales.scale([offr(cid)+1:offr(cid+1) offp(cid)+1:offp(cid+1)]);
    
    % Generates graphs.
    parfor vid = 1:length(video_list)
        % Loads the space-time segments.
        A = load([stsegment_path filesep num2str(vid) '_actionlet2.mat']);
        is_root = ([A.A(:).isRoot] > 0);
        stsegments = [A.A(is_root) A.A(~is_root)];
        
        % Generates graph for the original video.
        F1 = load([stsegment_path filesep num2str(vid) '_feat222.mat']);
        F2 = load([stsegment_path filesep num2str(vid) '_feat333.mat']);
        Fr = F2.F(is_root, 487:end); Fp = F1.F(~is_root, 145:end);
        Fr = normalize_feat(Fr, min_rf, max_rf);
        Fp = normalize_feat(Fp, min_pf, max_pf);
    
        tic; [RS, PS, G, GT, GS] = gengraph_hsts(stsegments, Fr, Fp, ...
            root_model, part_model, scale, graph_params); toc;
        save_graph([graph_path filesep num2str(cid) '_' num2str(vid) '_graph.mat'], ...
            RS, PS, G, GT, GS);
        
        % Generates graph for the flipped video.
        if used_for_training(vid) > 0
            F_flip = load([stsegment_path filesep num2str(vid) '_flip_feat.mat']);
            Fr = F_flip.fFRoot(:, 487:end); Fp = F_flip.fFPart(:, 145:end);
            Fr = normalize_feat(Fr, min_rf, max_rf);
            Fp = normalize_feat(Fp, min_pf, max_pf);
    
            tic; [RS, PS, G, GT, GS] = gengraph_hsts(stsegments, Fr, Fp, ...
                root_model, part_model, scale, graph_params); toc;
            save_graph([graph_path filesep num2str(cid) '_' num2str(vid) '_graph_flip.mat'], ...
                RS, PS, G, GT, GS);
        end
    end
end