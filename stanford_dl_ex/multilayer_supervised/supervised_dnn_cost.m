function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
for l = 1 : numHidden
    if l == 1
        z = stack{l}.W * data + stack{l}.b;
    else
        z = stack{l}.W * hAct{l - 1} + stack{l}.b;
    end
    hAct{l} = sigmoid(z);
%     hAct{1} is a(2) = f(z(2)) = f(theta(1)'a(1)), hAct{2} is a(3)
end
% calculate output layer
zOut = stack{numHidden + 1}.W * hAct{numHidden} ...
    + stack{numHidden + 1}.b;
hOut = zOut;

expT = exp(hOut);
sumExpT = sum(expT, 1);
% expT / sumExpT for all terms
hAct{numHidden + 1} = bsxfun(@rdivide, expT, sumExpT);
pred_prob = hAct{numHidden + 1};

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
%   take the class label for each column 
I = sub2ind(size(expT), labels', 1:size(expT,2));
% expClass = expT(I);
%   exp / expSum for selected term
indicatorP = pred_prob(I);
% indicatorP = bsxfun(@rdivide, expClass, sumExpT);
ceCost = -sum(log(indicatorP));



%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
% get error for the output layer
desired = zeros(size(pred_prob));
desired(I) = 1;
delta = pred_prob - desired;

for l = numHidden + 1: -1 : 2
    gradStack{l}.W = delta * hAct{l - 1}';
    gradStack{l}.b = sum(delta, 2);
    delta = (stack{l}.W)' * delta .* hAct{l - 1} .* ...
        (ones(size(hAct{l - 1})) - hAct{l - 1});
end
gradStack{1}.W = delta * data';
gradStack{1}.b = sum(delta, 2);

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for l = 1 : numHidden + 1
    wCost = wCost + 0.5 * ei.lambda * sum((stack{l}.W(:)).^2);
%     don't regularize softmax layer
    if l ~= (numHidden + 1)
        gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;
    end
end
cost = ceCost + wCost;


%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



