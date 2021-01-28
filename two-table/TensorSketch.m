function B = TensorSketch(A1, h1, A2, h2, m)
%
% Tensor Sketch on A1 \otimes A2

n1 = size(A1, 1);
d1 = size(A1, 2);
n2 = size(A2, 1);
d2 = size(A2, 2);
B = zeros(m, d1 + d2 - 1);
B1 = zeros(m, d1);
B2 = zeros(m, d2);
for i = 1: n1
    B1(h1(i), :) = B1(h1(i), :) + A1(i, :);
end
for i = 1: n2
    B2(h2(i), :) = B2(h2(i), :) + A2(i, :);
end
for i = 1: m
    for j = 1: m
        k = mod(i + j - 2, m) + 1;
        B(k, :) = B(k, :) + B1(i, :) * B2(j, :);
    end
end