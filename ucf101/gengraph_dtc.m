function G = gengraph_dtc(clusters, params)
% Generates graphs among dense trajectory clusters.

% Makes cluster id start from 1 and consecutive.
gidx = clusters.gidx(:);
gids = unique(gidx);
map(gids) = 1:length(gids);
gidx = map(gidx); gidx = gidx(:);
n = length(gids);

% Computes the mean temporal and spatial positions of each dt cluster.
t = accumarray(gidx, clusters.X(:, 1), [], @mean);
x = accumarray(gidx, clusters.X(:, 2), [], @mean);
y = accumarray(gidx, clusters.X(:, 3), [], @mean);

% Computes the space time edge labels among the clusters.
G = zeros(n);
for i = 1:n
    idx = find(abs(t - t(i)) <= params.time_diff_thresh);
    d = (x(idx) - x(i)).^2 + (y(idx) - y(i)).^2;
    [~, ix] = sort(d, 'ascend'); idx = idx(ix);
    idx1 = setdiff(idx(1 : min(length(idx), params.K + 1)), i);
    if ~isempty(idx1)
        for j = 1:length(idx1)
            id = idx1(j);
            if t(i) < t(id)
                [G(i, id), G(id, i)] = connect_node(...
                    t(i), y(i), t(id), y(id), params);
            else
                [G(id, i), G(i, id)] = connect_node(...
                    t(id), y(id), t(i), y(i), params);
            end
        end
    end
end

end

function [e12, e21] = connect_node(t1, y1, t2, y2, params)
% Computes the spatial temporal edge label between two nodes.
% Note this function assumes that t1 is smaller than t2. 
if t2 - t1 <= params.time_overlap_thresh
    if abs(y1 - y2) <= params.space_overlap_thresh
        e12 = params.OVERLAP_OVERLAP;
        e21 = e12 * -1;
    elseif y1 < y2
        e12 = params.OVERLAP_ABOVE;
        e21 = e12 * -1;
    else
        e21 = params.OVERLAP_ABOVE;
        e12 = e21 * -1;
    end
else
    if abs(y1 - y2) <= params.space_overlap_thresh
        e12 = params.BEFORE_OVERLAP;
        e21 = e12 * -1;
    elseif y1 < y2
        e12 = params.BEFORE_ABOVE;
        e21 = e12 * -1;
    else
        e12 = params.BEFORE_BELOW;
        e21 = e12 * -1;
    end
end

end