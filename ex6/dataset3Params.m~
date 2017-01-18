function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
temp_C = 0;
temp_sigma = 0;
lowest_error = 10000;
test_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
tol = 0.1;
max_passes = 100;
num = 0;

for i = 1:length(test_vals)
    for j = 1:length(test_vals)

        temp_C = test_vals(i);
        temp_sigma = test_vals(j);

        model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma)); 

        % Compute error
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));

        % if error < lowest_error, C = temp_C and sigma=temp_sigma
        if (prediction_error < lowest_error)
            C = temp_C;
            sigma = temp_sigma;
            lowest_error = prediction_error;
        endif

%        num = num + 1  
%        C
%        sigma
%        prediction_error
%        lowest_error
    endfor
endfor





% =========================================================================

end
