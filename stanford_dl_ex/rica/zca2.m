function [Z] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%
%% Step 0b: Zero-mean the data (by row)
%  You can make use of the mean and repmat/bsxfun functions.

%%% YOUR CODE HERE %%%
avg = mean(x, 1);
x = x - repmat(avg, size(x, 1), 1);

%% Step 1a: Implement PCA to obtain xRot
%  Implement PCA to obtain xRot, the matrix in which the data is expressed
%  with respect to the eigenbasis of sigma, which is the matrix U.

%%% YOUR CODE HERE %%%
sigma = x * x' / size(x, 2);
[U, S, V] = svd(sigma);

%% Step 4a: Implement PCA with whitening and regularisation
%  Implement PCA with whitening and regularisation to produce the matrix
%  xPCAWhite. 
xPCAwhite = diag(1./sqrt(diag(S) + epsilon)) * U' * x;

%% Step 5: Implement ZCA whitening
%  Now implement ZCA whitening to produce the matrix xZCAWhite. 
%  Visualise the data and compare it to the raw data. You should observe
%  that whitening results in, among other things, enhanced edges.

%%% YOUR CODE HERE %%%
Z = U * xPCAwhite;
