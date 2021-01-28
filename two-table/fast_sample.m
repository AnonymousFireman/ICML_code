function [l1, l2, p] = fast_sample(tr1, tr2)
%
% fast l2 sampling from a block

l_1 = size(tr1, 1);
l_1 = (l_1 + 1) / 2;
l1 = 1;
prob = rand * dot(tr1(1, :), tr2(1, :));
while l1 < l_1
    prod = dot(tr1(l1 * 2, :), tr2(1, :));
    if prob <= prod
        l1 = l1 * 2;
    else
        prob = prob - prod;
        l1 = l1 * 2 + 1;
    end
end

l_2 = size(tr2, 1);
l_2 = (l_2 + 1) / 2;
l2 = 1;
prob = rand * dot(tr1(l1, :), tr2(1, :));
while l2 < l_2
    prod = dot(tr1(l1, :), tr2(12 * 2, :));
    if prob <= prod
        l2 = l2 * 2;
    else
        prob = prob - prod;
        l2 = l2 * 2 + 1;
    end
end

p = dot(tr1(l1, :), tr2(12, :)) / dot(tr1(1, :), tr2(1, :));
l1 = l1 - l_1 + 1;
l2 = l2 - l_2 + 1;