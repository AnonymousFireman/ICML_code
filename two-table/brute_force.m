function A = brute_force(T1, T2)
%
% computing A = J^T J where J = join(T1, T2) by brute force

n1 = size(T1, 1);
n2 = size(T2, 1);
d1 = size(T1, 2);
d2 = size(T2, 2);

d = d1 + d2 - 1;
A = zeros(d, d);
mult = zeros(n1, 1);
l = 1;
r = 1;
for i = 1: n1
    while l <= n2 && T2(l, 1) < T1(i, d1)
        l = l + 1;
    end
    while r <= n2 && T2(r, 1) <= T1(i, d1)
        r = r + 1;
    end
    mult(i) = r - l;
end

for i = 1: d1
    for j = i: d1
        A(i, j) = A(i, j) + dot(T1(:, i) .* T1(:, j), mult);
    end
end

A(d1, d1) = 0;

mult = zeros(n2, 1);
l = 1;
r = 1;
for i = 1: n2
    while l <= n1 && T1(l, d1) < T2(i, 1)
        l = l + 1;
    end
    while r <= n1 && T1(r, d1) <= T2(i, 1)
        r = r + 1;
    end
    mult(i) = r - l;
end

for i = 1: d2
    for j = i: d2
        A(d1 + i - 1, d1 + j - 1) = A(d1 + i - 1, d1 + j - 1) + dot(T2(:, i) .* T2(:, j), mult);
    end
end

for i = 1: d1 - 1
    for j = 2: d2
        l = 1;
        r = 1;
        s = 0;
        for k = 1: n1
            while r <= n2 && T2(r, 1) <= T1(k, d1)
                s = s + T2(r, j);
                r = r + 1;
            end
            while l <= n2 && T2(l, 1) < T1(k, d1)
                s = s - T2(l, j);
                l = l + 1;
            end
            A(i, j + d1 - 1) = A(i, j + d1 - 1) + T1(k, i) * s;
        end
    end
end
for i = 1: d
    for j = 1: i - 1
        A(i, j) = A(j, i);
    end
end