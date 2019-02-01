function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

%%

h = zeros(m);
h = sigmoid(X*theta);
theta_j = theta([2:end],:);
J = (-y'*log(h) - (1-y')*log(1 - h))/m + (theta_j'*theta_j)*lambda/(2*m);
s = [0; lambda*theta_j];
grad = (X'*(h - y))/m + s/m;

% =============================================================

grad = grad(:);

end
