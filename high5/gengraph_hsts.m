function [G, GT, GS] = gengraph_hsts(S, params)        
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
% Author: Shugao Ma, 8/20/2014

n = length(S);
LARGE_NUM = 10^4;

% Computes the overlap ratios and distances among the segments.
R = zeros(n);
T = zeros(n);
D = ones(n) * LARGE_NUM;
Dy = D;
for i = 1 : n - 1 
    for j = i + 1 : n
        if S(i).isRoot == S(j).isRoot
            R(i, j) = compute_overlap(S(i), S(j));
            R(j, i) = R(i, j);
        end
    
        T(i, j) = min(S(i).end, S(j).end) - max(S(i).start, S(j).start);
        T(j, i) = T(i, j);
        
        if T(i, j) > params.tovlp_thre
            [D(i, j), Dy(i, j)] =  compute_distance(S(i), S(j));
            D(j, i) = D(i, j);
            Dy(j, i) = -1 * Dy(i, j);
        end
    end
end

% Computes graph edge labels. Each segment is connected to the top K
% nearest non-redundant segments.
G = zeros(n); GT = zeros(n); GS = zeros(n);
for i = 1 : n
    idx = setdiff(find(R(i, :) <= params.redundant_thre & ...
        T(i, :) > params.tovlp_thre), i);
    a = params.neighbor_K - length(idx);
    
    if a > 0
        idx1 = setdiff(find(R(i, :) <= params.redundant_thre & ...
            T(i, :) <= params.tovlp_thre), i);
        [~, ix] = sort(abs([S(idx1).start] - S(i).start), 'ascend'); 
        idx = [idx idx1(ix(1 : min(length(idx1), a)))];
    elseif a < 0
        [~, ix] = sort(D(i, idx), 'ascend');
        idx = idx(ix(1 : params.neighbor_K));
    end
    
    for z = 1 : length(idx)
        j = idx(z);
        [G(i, j), G(j, i), GT(i, j), GT(j, i), GS(i, j), GS(j, i)] = ...
            connect_nodes(S(i), S(j), T(i, j), Dy(i, j), params);
    end
end

end

%==========================================================================
% Computes overlap among segments.
%==========================================================================
function r = compute_overlap(s1, s2)
volumn_s1 = sum(s1.bbox(:, 3) .* s1.bbox(:, 4));
volumn_s2 = sum(s2.bbox(:, 3) .* s2.bbox(:, 4));
tstart = max(s1.start, s2.start);
tend = min(s1.end, s2.end);

if tend - tstart >= 1
    s1_com_bbox = s1.bbox((tstart - s1.start + 1) : ...
        (tstart - s1.start + 1) + (tend - tstart), :);
    s2_com_bbox = s2.bbox((tstart - s2.start + 1) : ...
        (tstart - s2.start + 1) + (tend - tstart), :); 
    volumn_int = sum(rectint2(s1_com_bbox, s2_com_bbox));
else
    volumn_int = 0;
end

r = volumn_int / (volumn_s1 + volumn_s2 - volumn_int);
end

%==========================================================================
% Computes the spatial distance of two overlapping segments.
%==========================================================================
function [d, dy] = compute_distance(s1, s2)
tstart = max(s1.start, s2.start);
tend = min(s1.end, s2.end);
s1_com_bbox = s1.bbox((tstart - s1.start + 1) : ...
    (tstart - s1.start + 1) + (tend - tstart), :);
s2_com_bbox = s2.bbox((tstart - s2.start + 1) : ...
    (tstart - s2.start + 1) + (tend - tstart), :);
y1 = mean(s1_com_bbox(1, 2) + s1_com_bbox(1, 4) / 2, 1);
y2 = mean(s2_com_bbox(1, 2) + s2_com_bbox(1, 4) / 2, 1);
x1 = mean(s1_com_bbox(1, 1) + s1_com_bbox(1, 3) / 2, 1);
x2 = mean(s2_com_bbox(1, 1) + s2_com_bbox(1, 3) / 2, 1);
dx = x1 - x2;
dy = y1 - y2;
d = sqrt(dx^2 + dy^2);
end

%==========================================================================
% Computes the edge label between two segments.
%==========================================================================
function [e12, e21, t12, t21, s12, s21] = connect_nodes(...
    s1, s2, tvlp, dy, params)

e12 = 0; e21 = 0;
t12 = 0; t21 = 0;
s12 = 0; s21 = 0;
swapped = 0;

% Swaps s1 and s2 so that s1 always start earlier.
if s1.start > s2.start
    tmp = s1;
    s1 = s2;
    s2 = tmp;
    dy = -1 * dy;
    swapped = 1;
end

% No edges between parts or part and root of different tracks.
if s1.tid ~= s2.tid && (s1.isRoot < 1 || s2.isRoot < 1)
    return;
end

% Assigns label to edges.
if tvlp < params.tovlp_thre
    e12 = params.BEFORE;
    e21 = -1 * params.BEFORE;
    t12 = 1; t21 = -1;
    s12 = 0; s21 = 0;
else
    if abs(dy) <= params.sabove_thre
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
        if dy < 0
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
