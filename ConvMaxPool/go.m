% Add the path to the halide tools.
addpath(fullfile(pwd, '..', 'halide', 'tools'));
% Build the halide pipeline into a mex library.
%mex_halide('ConvMaxPool.cpp');

% Some basic parameters of our filter bank and pooling.
rows = 300*8;
cols = 300*8;
filter_size = 5;
filter_count = 64;
pool_size = 4;

% Generate random data to use for testing.
in = single(randn(rows, cols));
filters = single(randn(filter_size, filter_size, filter_count));
out = zeros(rows, cols, filter_count);
%out = -inf(ceil(rows/pool_size), ceil(cols/pool_size), filter_count);

tic;
for c = 1:filter_count
    in_filter_c = conv2(in, filters(:,:,c), 'same'); 
    
    out(:,:,c) = in_filter_c;
    %for i = 1:pool_size
    %    for j = 1:pool_size
    %        out(:,:,c) = ...
    %            max(out(:,:,c), in_filter_c(i : pool_size : end, j : pool_size : end));
    %    end
    %end
end
toc;

%out_halide = zeros(size(out), 'single');

%tic;
%ConvMaxPool(in, filters, pool_size, out_halide);
%toc;

%max(max(max(abs(out - out_halide))))
