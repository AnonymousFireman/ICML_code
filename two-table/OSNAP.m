function B = OSNAP(A, m, s)
%
% Apply OSNAP to A with target dimension m and sparsity s

n = size(A, 1);
d = size(A, 2);
B = zeros(m, d);
S = zeros(n, s);
sig = randi(2, n, s) * 2 - 3;
% for i = 1: n
%     S(i, :) = randsample(m, s);
% end

S = randi(m, n, s);

B = osnap(A, m, s, S, sig);
% if issparse(A)
%     [p, q, r] = find(A);
%     nz = size(p, 1);
%     for l = 1: nz
%         i = p(l);
%         j = q(l);
%         v = r(l);
%         for k = 1: s
%             B(S(i, k), j) = B(S(i, k), j) + sig(i, k) * v;
%         end
%     end
% else
%     for i = 1: n
%         for j = 1: d
%             %B(S(i, :), j) = B(S(i, :), j) + sig(i, :)' .* A(i, j);
%             for k = 1: s
%                 B(S(i, k), j) = B(S(i, k), j) + sig(i, k) * A(i, j);
%             end
%         end
%     end
% end

B = B ./ sqrt(s);
