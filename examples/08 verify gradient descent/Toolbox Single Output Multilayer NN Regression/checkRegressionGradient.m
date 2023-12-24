% This program checks the correctness of the computation of the gradient of
% an artificial neural network for regression

% Escuela Politecnica Nacional
% Marco E. Benalcázar Palacios
% marco.benalcazar@epn.edu.ec
clc;
close all;
clear all;
warning off all;

N = 1000;
n = 35;
mean = 0;
sigma = 0.15;
numNeuronsLayers = [n 20 15 10 5 3 1];
transferFunctions{1} = 'none';
transferFunctions{2} = 'relu';
transferFunctions{3} = 'tanh';
transferFunctions{4} = 'softplus';
transferFunctions{5} = 'elu';
transferFunctions{6} = 'logsig';
transferFunctions{7} = 'elu';
options.lambda = 10;
options.reluThresh = 1e-2;

% Generating a training sample randomly
dataX = randn(N, n);
dataY = randi([1 100], N, 1).*randn(N, 1);

% Generating the weights of the neural network randomly
theta = [];
for i = 2:length(numNeuronsLayers)
    W = normrnd(mean, sigma, numNeuronsLayers(i), numNeuronsLayers(i - 1) + 1);
    theta = [theta; W(:)];
end

% Computing the gradients numerically
costFunction = @(t) regressionNNCostFunction(dataX, dataY,...
                                            numNeuronsLayers,...
                                            t,...
                                            transferFunctions,...
                                            options);
numericalGradient = computeNumericalGradient( costFunction, theta );

% Computing the exact gradients
[dummyVar, analyticalGradient] = regressionNNCostFunction(dataX, dataY,...
                                            numNeuronsLayers,...
                                            theta,...
                                            transferFunctions,...
                                            options);
                                        
% Comparing numerically computed gradients with those computed analytically
diff = norm(numericalGradient - analyticalGradient)/...
       norm(numericalGradient + analyticalGradient);
disp([numericalGradient analyticalGradient]);  
fprintf('\n difference = %d\n', diff);
% This value is usually less than 1e-7                                        