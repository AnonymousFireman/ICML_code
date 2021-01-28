function B = get_block(T1, T2)
%
% Finding all blocks

n1 = size(T1, 1);
d1 = size(T1, 2);
n2 = size(T2, 1);
d2 = size(T2, 2);
B = zeros(n1, 4);
nb = 0;
l1 = 1;
l2 = 1;
while l1 <= n1 && l2 <= n2
    while l2 <= n2 && T2(l2, 1) < T1(l1, d1)
        l2 = l2 + 1;
    end
    r2 = l2;
    while r2 <= n2 && T2(r2, 1) <= T1(l1, d1)
        r2 = r2 + 1;
    end
    r1 = l1;
    while r1 <= n1 && T1(r1, d1) == T1(l1, d1)
        r1 = r1 + 1;
    end
    if r2 > l2
        nb = nb + 1;
        B(nb, 1) = l1;
        B(nb, 2) = r1;
        B(nb, 3) = l2;
        B(nb, 4) = r2;
    end
    l1 = r1;
end
B = B(1: nb, :);