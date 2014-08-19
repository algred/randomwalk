function [RS, PS, G, GT, GS] = gengraph_hsts(S, Fr, Fp, ...
            root_model, part_model, scale, params)
        
% Generate a space-time graph given a set of space-time segments of a
% video.
% Input:
%   S: S(i) is a struct of a space-time segments. The first m1 stsegments
%      are roots and the next m2 are parts.
%   Fr: m1 * root_dim matrix of root stsegment features.
%   Fp: m2 * part_dim matrix of part stsegment features.
%   root_model, part_model: dsc cluster models.
%   scale: for normalization.
% Output:
%   RS, PS: root and part STS's scores over codebook.
%   G: G(i, j) stores the space-time edge label between space-time segment
%      from i to j.
%   GT, GS: separated time and space edge matrix. Values for ET(i, j) are
%           BEFORE (1), AFTER(-1), OVERLAP(2), NO-EDGE(0).
%           Values for ES(i, j) are ABOVE(1), BELOW(-1), OVERLAP(2),
%           NO-EDGE(0).
%
% Edge labels:
%   params.BEFORE, params.OVERLAP, params.ABOVE
%
% Note in this version:
%
% 1) Each node should not be above / below / overlap / befor / after more than
%    1 other nodes that have the same label.
% 2) No edges between parts that are not overlap in time.
% 3) BEFORE is measured by the starting frame of a space-time segment.
% 4) Time label is always assigned to an edge.
%
% Author: Shugao Ma, 1/21/2014
%
% Difference with graphgen6
% 1. params.top_label_num is not used to limit the possible labels a
%    space-time segment.
%
% 2. Each node is allowed to be above / below / overlap / befor / after
%    more than 1 other nodes that have the same label.
%
%3. The output is different to save space.

n = length(S);
RS = []; PS = [];

if ~isempty(Fr)
    RS = Fr * root_model.Ws + repmat([root_model.bs(:)]', size(Fr, 1), 1);
    RS = exp(RS) ./ repmat(scale(1:size(RS, 2)), size(RS, 1), 1);
end

if ~isempty(Fp)
    PS = Fp * part_model.Ws + repmat([part_model.bs(:)]', size(Fp, 1), 1);
    PS = exp(PS) ./ repmat(scale(size(RS, 2)+1:end), size(PS, 1), 1);
end

for i = 1:n-1
  for j = i+1:n
    [G(i, j), G(j, i), GT(i, j), GT(j, i), GS(i, j), GS(j, i)] = connect_nodes(S(i), S(j), params);
  end
end

end

function [e12, e21, t12, t21, s12, s21] = connect_nodes(s1, s2, params)
e12 = 0; e21 = 0;
t12 = 0; t21 = 0;
s12 = 0; s21 = 0;
swapped = 0;

% Swaps s1 and s2 so that s1 always start earlier.
if s1.start > s2.start
    tmp = s1;
    s1 = s2;
    s2 = tmp;
    swapped = 1;
end

% No edges between parts or part and root of different tracks.
if s1.tid ~= s2.tid && (s1.isRoot < 1 || s2.isRoot < 1)
    return;
end

% No edges between redundant space-time segments.
volumn_s1 = sum(s1.bbox(:, 3) .* s1.bbox(:, 4));
volumn_s2 = sum(s2.bbox(:, 3) .* s2.bbox(:, 4));

tstart = max(s1.start, s2.start);
tend = min(s1.end, s2.end);
if tend - tstart >= 1
    s1_com_bbox = s1.bbox((tstart - s1.start + 1) : (tstart - s1.start + 1) + (tend - tstart), :);
    s2_com_bbox = s2.bbox((tstart - s2.start + 1) : (tstart - s2.start + 1) + (tend - tstart), :);
    volumn_int = sum(diag(rectint(s1_com_bbox, s2_com_bbox)));
else
    volumn_int = 0;
end

if volumn_int / (volumn_s1 + volumn_s2 - volumn_int) > params.redundant_thre
    return;
end

% Assigns label to edges.
if (tend - tstart) < params.tovlp_thre
    % Assigns only time label if two segments are not significantly overlap in time.
    e12 = params.BEFORE;
    e21 = -1 * params.BEFORE;
    t12 = 1; t21 = -1;
    s12 = 0; s21 = 0;
else
    % Assigns both time and space label if s1 and s2 overlap in time and
    % their start times are significantly different.
    y1 = mean(s1_com_bbox(1, 2) + s1_com_bbox(1, 4) / 2, 1);
    y2 = mean(s2_com_bbox(1, 2) + s2_com_bbox(1, 4) / 2, 1);
    if abs(y1 - y2) <= params.sabove_thre
        if s2.start - s1.start >= params.tbefore_thre
            e12 = params.BEFORE_OVERLAP;
            e21 = -1 * params.BEFORE_OVERLAP;
            t12 = 1; t21 = -1;
            s12 = 2; s21 = 2;
        else
            e12 = params.OVERLAP;
            e21 = params.OVERLAP;
            t12 = 2; t21 = 2;
            s12 = 2; s21 = 2;
        end
    else
        if y1 < y2
            if s2.start - s1.start >= params.tbefore_thre
                e12 = params.BEFORE_ABOVE;
                e21 = -1 * params.BEFORE_ABOVE;
                t12 = 1; t21 = -1;
                s12 = 1; s21 = -1;
            else
                e12 = params.ABOVE;
                e21 = -1 * params.ABOVE;
                t12 = 2; t21 = 2;
                s12 = 1; s21 = -1;
            end
        else
            if s2.start - s1.start >= params.tbefore_thre
                e12 = params.BEFORE_BELOW;
                e21 = -1 * params.BEFORE_BELOW;
                t12 = 1; t21 = -1;
                s12 = -1; s21 = 1;
            else
                e12 = -1 * params.ABOVE;
                e21 = params.ABOVE;
                t12 = 2; t21 = 2;
                s12 = 1; s21 = -1;
            end
        end
    end
end

% If s1 and s2 are swapped, we also need to swap the edge labels.
if swapped > 0
    tmp = e12; e12 = e21; e21 = tmp;
    tmp = t12; t12 = t21; t21 = tmp;
    tmp = s12; s12 = s21; s21 = tmp;
end

end
