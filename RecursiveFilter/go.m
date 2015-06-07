% Add the path to the Halide MATLAB tools.
addpath(fullfile(pwd, '..', 'halide', 'tools'));

% Build the blur function.
mex_halide('IirBlur.cpp');

% Load an input image.
in = single(imread('rgb.png')) / 255;
% Create an output buffer for the result.
out = zeros(size(in), 'single');
% This defines the amount of blur (the first order IIR filter coefficient).
A = 0.05;

% Time how long it takes to run the blur 10 times.
tic;
for i = 1:10
    IirBlur(in, A, out);
end
toc;

imshow(out);


