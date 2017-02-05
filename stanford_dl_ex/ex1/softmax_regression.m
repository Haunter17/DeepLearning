function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  theta_temp = theta;
  theta = [theta, zeros(n, 1)];
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  
  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  for i = 1 : m
      iSum = 0;
      expSum = 0;
      for j = 1 : num_classes
          expSum = expSum + exp(theta(:, j).' * X(:, i));
      end
      
      for k = 1 : num_classes
          iSum = iSum + (y(i) == k) * ...
              log(exp(theta(:, k).' * X(:,i)) / expSum);
      end
      f = f - iSum;
  end
  
  for k = 1 : num_classes
      accum = zeros(size(X, 1), 1);
      for i = 1 : m
          expSum = 0;
          for j = 1 : num_classes
              expSum = expSum + exp(theta(:, j).' * X(:, i));
          end
          diff = (y(i) == k) - exp(theta(:,k).' * X(:, i)) / expSum;
          accum = accum + X(:, i) * diff;
      end
      g(:, k) = -accum;
  end
  g = g(:, 1 : num_classes - 1);
  
  
  g=g(:); % make gradient a vector for minFunc

