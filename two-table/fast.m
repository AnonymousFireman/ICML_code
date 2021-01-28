function y = fast(T1, T2, x, B)
%
% computing y = J^T*J*x

n1 = size(T1, 1);
d1 = size(T1, 2);
n2 = size(T2, 1);
d2 = size(T2, 2);
d = d1 + d2 - 1;
y = zeros(d, 1);
n_B = size(B, 1);
for i = 1: n_B
    l1 = B(i, 1);
    r1 = B(i, 2);
    l2 = B(i, 3);
    r2 = B(i, 4);
    t1 = T1(l1: r1 - 1, :) * x(1: d1);
    t2 = T2(l2: r2 - 1, 2: d2) * x(d1 + 1: d);
    %y(1: d1) = y(1: d1) + T1(l1: r1 - 1, :)' * t1 .* (r2 - l2);
    %y(d1 + 1: d) = y(d1 + 1: d) + sum(t1) .* sum(T2(l2: r2 - 1, 2: d2), 1)';
    %y(1: d1) = y(1: d1) + sum(T1(l1: r1 - 1, :), 1)' .* sum(t2);
    %y(d1 + 1: d) = y(d1 + 1: d) + T2(l2: r2 - 1, 2: d2)' * t2 .* (r1 - l1);
    y(1: d1) = y(1: d1) + T1(l1: r1 - 1, :)' * (t1 .* (r2 - l2) + sum(t2));
    y(d1 + 1: d) = y(d1 + 1: d) + T2(l2: r2 - 1, 2: d2)' * (t2 .* (r1 - l1) + sum(t1));
end