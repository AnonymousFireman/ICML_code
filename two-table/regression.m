function y = regression(T1, T2, x)
%
% computing y = |Jx|^2

l1 = 1;
r1 = 1;
l2 = 1;
r2 = 1;
n1 = size(T1, 1);
d1 = size(T1, 2);
n2 = size(T2, 1);
d2 = size(T2, 2);
d = d1 + d2 - 1;
y = 0;
while l1 <= n1 && l2 <= n2
    while l2 <= n2 && T2(l2, 1) < T1(l1, d1)
        l2 = l2 + 1;
    end
    while r2 <= n2 && T2(r2, 1) <= T1(l1, d1)
        r2 = r2 + 1;
    end
    while r1 <= n1 && T1(r1, d1) == T1(l1, d1)
        r1 = r1 + 1;
    end
    if r2 > l2
        t1 = T1(l1: r1 - 1, :) * x(1: d1);
        t2 = T2(l2: r2 - 1, 2: d2) * x(d1 + 1: d);
        y = y + (r2 - l2) * sum(t1 .^2);
        y = y + (r1 - l1) * sum(t2 .^2);
        y = y + 2 * sum(t1) * sum(t2);
    end
    l1 = r1;
end