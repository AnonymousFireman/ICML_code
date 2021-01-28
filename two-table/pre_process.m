function [tr1, tr2] = pre_process(T1, T2, Y, B1, B2, nb)
%
% pre-processing for l_2 sampling from rows of J*Y

d = size(Y, 1);
r = size(Y, 2);
n1 = size(T1, 1);
d1 = size(T1, 2);
n2 = size(T2, 1);
d2 = size(T2, 2);
tr1 = cell(nb, 1);
tr2 = cell(nb, 1);

for i = 1: nb
    j = B1(i) + 1;
    k = B1(i + 1);
    l = k - j + 1;
    tr1{i} = zeros(2 * l - 1, 3 * r);
    tr1{i}(l: 2 * l - 1, 1: r) = 1;
    tr1{i}(l: 2 * l - 1, r + 1: 2 * r) = T1(j: k, :) * Y(1: d1, :);
    tr1{i}(l: 2 * l - 1, 2 * r + 1: 3 * r) = tr1{i}(l: 2 * l - 1, r + 1: 2 * r).^2;
    
    tr1{i}(l: 2 * l - 1, r + 1: 2 * r) = tr1{i}(l: 2 * l - 1, r + 1: 2 * r) * 2;
    for t = l - 1: -1: 1
        tr1{i}(t, :) = tr1{i}(t * 2, :) + tr1{i}(t * 2 + 1, :);
    end
end

for i = 1: nb
    j = B2(i) + 1;
    k = B2(i + 1);
    l = k - j + 1;
    tr2{i} = zeros(2 * l - 1, 3 * r);
    tr2{i}(l: 2 * l - 1, 1: r) = 1;
    tr2{i}(l: 2 * l - 1, r + 1: 2 * r) = T2(j: k, 2: d2) * Y(d1 + 1: d, :);
    tr2{i}(l: 2 * l - 1, 2 * r + 1: 3 * r) = tr2{i}(l: 2 * l - 1, r + 1: 2 * r).^2;
    
    for t = l - 1: -1:  1
        tr2{i}(t, :) = tr2{i}(t * 2, :) + tr2{i}(t * 2 + 1, :);
    end
end
