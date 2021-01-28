function C = TensorSRHT(A, B, m)
%
% Apply TensorSRHT to A \tensor B with target dimension m

n1 = size(A, 1);
d1 = size(A, 2);
n2 = size(B, 1);
d2 = size(B, 2);
for i = 1: n1
    A(i, :) = A(i, :) .* (randi(2) * 2 - 3);
end
for i = 1: n2
    B(i, :) = B(i, :) .* (randi(2) * 2 - 3);
end
A = Hadamard8(A);
B = Hadamard8(B);
%A = ifwht(A, n1, 'hadamard');
%B = ifwht(B, n2, 'hadamard');
p = randi(m, m, 1);
q = randi(m, m, 1);
C = zeros(m, d1 * d2);
for i = 1: m
    C(i, :) = A(p(i), :) * B(q(i), :);
end
C = C ./ sqrt(m);