function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
z = X*theta;
h = 1./(1+exp(-z));
for i = 1:length(y)
  J = J + 1/m.*(-y(i)*log(h(i))-(1-y(i))*log(1-h(i)));
end
for i = 2:length(theta)
  J = J + lambda/(2*m)*theta(i).^2;
end
for i = 1:length(y)
    grad(1) = grad(1) + 1/m*(h(i)-y(i))*X(i,1);
end

for j = 2:length(theta)
  for i = 1:length(y)
    grad(j) = grad(j) + 1/m*(h(i)-y(i))*X(i,j);
  end
    grad(j) = grad(j) + lambda/(m)*theta(j);
end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
