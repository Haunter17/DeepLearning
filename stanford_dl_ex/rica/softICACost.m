%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
%% Compute Cost
temp1 = sqrt((W * x).^ 2 + params.epsilon);
regCost = params.lambda * sum(sum(temp1));
temp2 = (W' * W * x - x).^ 2;
RICACost = 0.5 * sum(sum(temp2)) / size(x, 2);
cost = regCost + RICACost;
%% Compute Gradient
RICAgrad = (W * (W' * W * x - x) * x' + (W * x) * (W' * W * x - x)')...
    / size(x, 2);
regGrad = params.lambda * W * x ./ temp1 * x';
Wgrad = regGrad + RICAgrad;

%%
% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);