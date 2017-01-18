function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp = 0;
J_i = 0;

for iter = 1:num_iters
    delta = zeros(length(theta),1);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % Calculate the delta
    for j = 1:length(theta)
        for i = 1:m
            delta(j) = delta(j) + ((X(i,:) * theta) - y(i)) * X(i,j)';
        end
    end

    % Fold back into theta
    for j = 1:length(theta)
        theta(j) = theta(j) - ((alpha) * (1 / m) * delta(j));
    end

    % Uncomment this to verify J(theta) is decreasing 
    %J_i = computeCost(X,y,theta)

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
