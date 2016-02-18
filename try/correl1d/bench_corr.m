% matlab -singleCompThread

n0 = 64;
n1 = 32;

disp(['n0: ', num2str(n0), ' ; n1: ', num2str(n1)]);

in0 = rand([n0, n0], 'single');
in1 = rand([n1, n1], 'single');

tic
norm = sum(sum(in1.^2));
r = conv2(in0, in1)./norm;
toc

tic
norm = sum(sum(in0.^2));
r = (real(ifft2(conj(fft2(in0)) .* fft2(in0))))./norm;
toc

% n0: 32 ; n1: 16
% Elapsed time is 0.000714 seconds.
% Elapsed time is 0.000314 seconds.

% n0: 64 ; n1: 32
% Elapsed time is 0.005700 seconds.
% Elapsed time is 0.000639 seconds.
