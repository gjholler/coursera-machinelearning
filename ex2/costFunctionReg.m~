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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%
% Calculate the Cost Function
delta1 = 0;
delta2 = 0;
h_theta = zeros(size(theta));

for i = 1:m
    h_theta = sigmoid(X(i,:) * theta);
    delta1 = delta1 + (-(y(i,:) * log(h_theta)) - ((1 - y(i,:)) * log(1 - h_theta)));
end

for j = 2:length(theta)
    delta2 = delta2 + theta(j) ^ 2;
end

J = ((1 / m) * delta1) + ((lambda / (2 * m)) * delta2);


%
% Calculate the gradient
delta = zeros(size(theta));
h_theta = zeros(size(theta));
grad = zeros(size(theta));

for j = 1:length(theta)
    for i = 1:m
        h_theta = sigmoid(X(i,:) * theta);
        delta(j) = delta(j) + (h_theta - y(i)) * X(i,j)';
    end
end

% Fold back into theta

% Handle theta(0)
grad(1) = (1 / m) * delta(1);

for j = 2:length(theta)
    grad(j) = ((1 / m) * delta(j)) + ((lambda / m) * theta(j));
end





% =============================================================

end
