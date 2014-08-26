init_high5;

% ==========================================================================
%
% Generates graphs.
%
% ==========================================================================
train_idx = find(used_for_training > 0);
load([model_path filesep 'cluster_scales.mat']);
ncls = 4;

% Loads the codebooks.
root_model = cell(ncls, 1);
part_model = cell(ncls, 1);
for cid = 1:ncls
    clusters = load([model_path filesep 'cluster_' num2str(cid) '.mat']);
    root_model{cid} = clusters.root_dscluster.model;
    part_model{cid} = clusters.part_dscluster.model;
end

for vid = vid1 : vid2
    % Loads the space-time segments.
    A = load([stsegment_path filesep num2str(vid) '_actionlet2.mat']);
    is_root = ([A.A(:).isRoot] > 0);
    stsegments = [A.A(is_root) A.A(~is_root)];
    
    % Generates graph for the original video.
    if ~exist([graph_path filesep num2str(vid) '_graph.mat'], 'file')
        F1 = load([stsegment_path filesep num2str(vid) '_feat222.mat']);
        F2 = load([stsegment_path filesep num2str(vid) '_feat333.mat']);
        Fr = F2.F(is_root, 487:end); Fp = F1.F(~is_root, 145:end);
        Fr = normalize_feat(Fr, min_rf, max_rf);
        Fp = normalize_feat(Fp, min_pf, max_pf);
        RS = cell(ncls, 1);
        PS = cell(ncls, 1);
        for cid = 1:ncls
            if ~isempty(Fr)
                RS{cid} = exp(Fr * root_model{cid}.Ws + ...
                    repmat([root_model{cid}.bs(:)]', size(Fr, 1), 1));
            end
            if ~isempty(Fp)
                PS{cid} = exp(Fp * part_model{cid}.Ws + ...
                    repmat([part_model{cid}.bs(:)]', size(Fp, 1), 1));
            end
        end
        
        fprintf('Finished generate graph for video %d\n', vid);
        tic; [G, GT, GS] = gengraph_hsts(stsegments, graph_params); toc;
        save_graph([graph_path filesep num2str(vid) '_graph.mat'], ...
            RS, PS, G, GT, GS);
    end
    
    % Generates graph for the flipped video.
    if used_for_training(vid) > 0 && ~exist(...
        [graph_path filesep num2str(vid) '_graph_flip.mat'], 'file')
        F_flip = load([stsegment_path filesep ...
            num2str(vid) '_flip_feat.mat']);
        Fr = F_flip.fFRoot(:, 487:end); Fp = F_flip.fFPart(:, 145:end);
        Fr = normalize_feat(Fr, min_rf, max_rf);
        Fp = normalize_feat(Fp, min_pf, max_pf);
        RS = cell(ncls, 1);
        PS = cell(ncls, 1);
        for cid = 1:ncls
            if ~isempty(Fr)
                RS{cid} = exp(Fr * root_model{cid}.Ws + ...
                    repmat([root_model{cid}.bs(:)]', size(Fr, 1), 1));
            end
            if ~isempty(Fp)
                PS{cid} = exp(Fp * part_model{cid}.Ws + ...
                    repmat([part_model{cid}.bs(:)]', size(Fp, 1), 1));
            end
        end
        
        fprintf('Finished generate fliped graph for video %d\n', vid);
        tic; [G, GT, GS] = gengraph_hsts(stsegments, graph_params); toc;
        save_graph([graph_path filesep num2str(vid) '_graph_flip.mat'], ...
            RS, PS, G, GT, GS);
    end
end