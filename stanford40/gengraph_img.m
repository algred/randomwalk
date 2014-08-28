function G = gengraph_img(X, params)
% Generates a graph among image windows.
%
% Input: 
%   X:       X(i, :) = [x y w h] is the window position.
%
% Output:
%   G:       G(i, j) contains the edge label between window i and j.

n = size(X, 1);
x = X(:, 1) + X(:, 3) / 2;
y = X(:, 2) + X(:, 4) / 2;

XX1 = repmat(x, 1, n);
XX2 = repmat(x', n, 1);
dx =  XX1 - XX2;

YY1 = repmat(y, 1, n);
YY2 = repmat(y', n, 1);
dy = YY1 - YY2; 

d = dx.^2 + dy.^2;
[~, ix] = sort(d, 2, 'ascend');

abs_dx = abs(dx);
abs_dy = abs(dy);
G = zeros(n);
for i = 1 : n
    idx = ix(i, 1 : min(n, (params.neighbor_K + 1)));
    idx = idx(idx ~= i);
    for j = 1 : length(idx)
        id = idx(j);
        if abs_dx(i, id) <= params.overlap_thresh
            if abs_dy(i, id) <= params.overlap_thresh
                G(i, id) = params.OVERLAP_OVERLAP;
                G(id, i) = params.OVERLAP_OVERLAP;
            elseif dy(i, id) < 0
                G(i, id) = params.OVERLAP_ABOVE;
                G(id, i) = -1 * G(i, id);
            elseif dy(i, id) > 0
                G(id, i) = params.OVERLAP_ABOVE;
                G(i, id) = -1 * G(id, i);
            end
        else
            if abs_dy(i, id) <= params.overlap_thresh
                G(i, id) = params.APART_OVERLAP;
                G(id, i) = params.APART_OVERLAP;
            elseif dy(i, id) < 0
                G(i, id) = params.APART_ABOVE;
                G(id, i) = -1 * G(i, id);
            elseif dy(i, id) > 0
                G(id, i) = params.APART_ABOVE;
                G(i, id) = -1 * G(id, i);
            end
        end
    end
end

end