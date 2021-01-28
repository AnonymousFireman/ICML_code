rng(1);

% J1
% fid = fopen('user_artists.dat');
% data = textscan(fid, '%f%f%f', 'headerlines', 1);
% fclose(fid);
% A = cat(2, data{:});
% A(:, [1, 3]) = A(:, [3, 1]);
% A = A - min(A);
% A = A ./ max(A);
% 
% fid = fopen('user_taggedartists-timestamps.dat');
% data = textscan(fid, '%f%f%f%f', 'headerlines', 1);
% fclose(fid);
% B = cat(2, data{:});
% B = B - min(B);
% B = B ./ max(B);

% J2
fid = fopen('ratings.dat');
data = textscan(fid, '%f%f%f%f', 'headerlines', 1);
fclose(fid);
A = cat(2, data{:});
A(:, [2, 4]) = A(:, [4, 2]);
A = A - min(A);
A = A ./ max(A);
fid = fopen('movies.dat');
data = textscan(fid, '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f', 'headerlines', 1);
fclose(fid);
B = cat(2, data{:});
B = B - min(B);
B = B ./ max(B);

A = sortrows(A, size(A, 2));
B = sortrows(B, 1);

X = calc_err(A, B, 0, 3); % 1
disp(X);
