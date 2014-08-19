function F = normalize_feat(F, min_f, max_f)
F = F - repmat(min_f, size(F, 1), 1);
F = F./repmat(max_f - min_f + eps, size(F, 1), 1);
F = F./repmat(sqrt(sum(F.^2, 2)), 1, size(F, 2));
end