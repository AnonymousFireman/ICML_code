function X = calc_err(T1, T2, lambda, j)
% X = calc_err(T1, T2, lambda, j)
%
% Computing the relative error between min_x |Ax - b|^2 + lambda |x|^2 and
% |Ax_0 - b|^2 + lambda |x_0|^2, where x_0 minimizes |SAx - Sb|^2 + lambda
% |x|^2, b is the j-th column of A = T_1 x T_2.

n1 = size(T1, 1);
n2 = size(T2, 1);
d1 = size(T1, 2);
d2 = size(T2, 2);
T1 = sortrows(T1, d1);
T2 = sortrows(T2, 1);
d = d1 + d2 - 1;
U = [1:j-1, j+1:d];

tic;
A = brute_force(T1, T2);

x = (A(U, U) + lambda .* eye(d-1)) \ (A(U, j));
toc;

%sig = svd(A);
%disp(sig);

tic;
B = get_block(T1, T2);
J = embedding(T1, T2, B);
disp(size(J));
JTJ = J' * J;

y = (JTJ(U, U) + lambda .* eye(d-1)) \ (JTJ(U, j));
% [Q, R] = qr(J(:, U), 0);
% y = R * y;
% R = inv(R);
% for i = 1: 0
%     tmp = R * y;
%     tmp = fast(T1, T2, [tmp(1: j-1); -1; tmp(j:d-1)], B);
%     y = y - R' * (tmp(U));
% end
% y = R * y;
toc;

x = [x(1: j-1); -1; x(j:d-1)];
y = [y(1: j-1); -1; y(j:d-1)];
ans1 = regression(T1, T2, x);
ans2 = regression(T1, T2, y);
disp([ans1, ans2]);
X = (ans2 - ans1) / ans1;