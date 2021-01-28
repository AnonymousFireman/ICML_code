function J = embedding(T1, T2, B)
%
% subspace embedding of T1 x T2

n1 = size(T1, 1);
d1 = size(T1, 2);
n2 = size(T2, 1);
d2 = size(T2, 2);
d = d1 + d2 - 1;

if issparse(T1)
    gamma = d * d;
else
    gamma = d; % 50 * d
end

%Tsmall1 = zeros(n1, d1);
%Tsmall2 = zeros(n2, d2);
J = zeros(20000, d);
B1 = zeros(n1, 2);
B2 = zeros(n2, 2);
sz = zeros(n1, 1);
f = zeros(n1, 1);
g = zeros(n2, 1);

n_B = size(B, 1);

ns1 = 0;
ns2 = 0;
nj = 0;
nb = 0;

for i = 1: n_B
    l1 = B(i, 1);
    r1 = B(i, 2);
    l2 = B(i, 3);
    r2 = B(i, 4);
    if r1 - l1 >= gamma || r2 - l2 >= gamma
        m = 4;
%         s = 1;
%         tmp1 = OSNAP([T1(l1: r1 - 1, 1: d1), ones(r1 - l1, 1)], m, s);
%         tmp2 = OSNAP([ones(r2 - l2, 1), T2(l2: r2 - 1, 2: d2)], m, s);
%         J(nj + 1: nj + m, 1: d1) = TensorSRHT(tmp1(:, 1: d1), tmp2(:, 1), m);
%         J(nj + 1: nj + m, d1 + 1: d) = TensorSRHT(tmp1(:, d1 + 1: d1 + 1), tmp2(:, 2: d2), m);
        h1 = randi(m, r1 - l1, 1);
        h2 = randi(m, r2 - l2, 1);
        sig1 = randi(2, r1 - l1, 1) * 2 - 3;
        sig2 = randi(2, r2 - l2, 1) * 2 - 3;
        J(nj + 1: nj + m, 1: d1) = tensorsketch(T1(l1: r1 - 1, 1: d1) .* sig1, h1, sig2, h2, m);
        J(nj + 1: nj + m, d1 + 1: d) = tensorsketch(sig1, h1, T2(l2: r2 - 1, 2: d2) .* sig2, h2, m);
        nj = nj + m;
    else
        %Tsmall1(ns1 + 1: ns1 + r1 - l1, :) = T1(l1: r1 - 1, :);
        %Tsmall2(ns2 + 1: ns2 + r2 - l2, :) = T2(l2: r2 - 1, :);
        nb = nb + 1;
        B1(nb, 1) = l1;
        B1(nb, 2) = r1 - 1;
        B2(nb, 1) = l2;
        B2(nb, 2) = r2 - 1;
        ns1 = ns1 + r1 - l1;
        ns2 = ns2 + r2 - l2;
        f(nb) = r1 - l1;
        g(nb) = r2 - l2;
        sz(nb) = (r1 - l1) * (r2 - l2);
    end
end

J = J(1: nj, :);
return;

if issparse(T1)
    m = ceil((ns1 + ns2) / d);
else
    m = ceil((ns1 + ns2) / 1000);
end
id = randsample(nb, m, true, sz(1: nb) / sum(sz(1: nb)));
k1 = zeros(m, 1);
k2 = zeros(m, 1);
for i = 1: m
   k1(i) = randi(f(id(i)));
   k2(i) = randi(g(id(i)));
end
k1(1: m) = k1(1: m) + B1(id(1: m), 1) - 1;
k2(1: m) = k2(1: m) + B2(id(1: m), 1) - 1;
Ju = zeros(m, d);
Ju(1: m, 1: d1) = T1(k1, 1: d1);
Ju(1: m, d1 + 1: d) = T2(k2, 2: d2);
t = d;
s = 1;
Ju = OSNAP(Ju, t, s);
[~, S, V] = svd(Ju, 'econ');

G = (eye(d) - V * V') * randn(d, 1);
for i = 1: nb
    a = T1(B1(i, 1): B1(i, 2), :) * G(1: d1);
    b = T2(B2(i, 1): B2(i, 2), 2: d2) * G(d1 + 1: d);
    gv = randn(1, f(i));
    p = gv * a + b .* sum(gv);
    for j = 1: g(i)
        if p(j) > 0.001
            for k = 1: f(i)
                if a(j) + b(k) > 0.001
                    nj = nj + 1;
                    J(nj, 1: d1) = T1(B1(i, 1) + k - 1, :);
                    J(nj, d1 + 1: d) = T2(B2(i, 1) + j - 1, 2: d2);
                end
            end
        end
    end
end

for i = 1: d
    if S(i, i) > 0
        S(i, i) = 1 / S(i, i);
    else
        S(i, i) = 0;
    end
end
t = 2;
G = V * S * randn(d, t);
if issparse(T1)
    alpha = ceil((d * d * d * d) );
else
    alpha = ceil((d * d) * 2);
end
a = zeros(n1, t);
b = zeros(n2, t);
c = zeros(n2, 1);
p = zeros(nb, 1);
for i = 1: nb
    gv = randn(t, f(i));
    for j = 1: t
        a(B1(i, 1): B1(i, 2), j) = T1(B1(i, 1): B1(i, 2), :) * G(1: d1, j);
        b(B2(i, 1): B2(i, 2), j) = T2(B2(i, 1): B2(i, 2), 2: d2) * G(d1 + 1: d, j);
        c(B2(i, 1): B2(i, 2)) = c(B2(i, 1): B2(i, 2)) + ((gv(j, :) * a(B1(i, 1): B1(i, 2), j) + b(B2(i, 1): B2(i, 2), j) * sum(gv(j, :))).^ 2);
    end
    p(i) = sum(c(B2(i, 1): B2(i, 2)));
    c(B2(i, 1): B2(i, 2)) = c(B2(i, 1): B2(i, 2)) / p(i);
end
p = p / sum(p);
id = randsample(nb, alpha, true, p);
for i = 1: alpha
    u = id(i);
    %k2 = randsample(g(u), 1, true, c(B2(u, 1): B2(u, 2)));
    k2 = choice(c(B2(u, 1): B2(u, 2)), rand);
    pt = zeros(f(u), 1);
    for j = 1: t
        pt = pt + (a(B1(u, 1): B1(u, 2), j) + b(B2(u, 1) + k2 - 1, j)).^ 2;
    end
    pt = pt / sum(pt);
    %k1 = randsample(f(u), 1, true, pt);
    k1 = choice(pt, rand);
    nj = nj + 1;
    J(nj, 1: d1) = T1(B1(u, 1) + k1 - 1, :);
    J(nj, d1 + 1: d) = T2(B2(u, 1) + k2 - 1, 2: d2);
    J(nj, :) = J(nj, :) / sqrt(p(u) * c(B2(u, 1) + k2 - 1) * pt(k1) * alpha);
end

J = J(1: nj, :);