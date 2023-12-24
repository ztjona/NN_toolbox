function initialTheta = randInitializeWeights(numNeuronsLayers)
% Initializing the Neural Network Parameters Randomly
mean = 0;
sigma = 0.5;
initialTheta = [];
for i = 2:length(numNeuronsLayers)   
    W = normrnd(mean, sigma, numNeuronsLayers(i), numNeuronsLayers(i - 1) + 1) / sqrt( numNeuronsLayers(i - 1) );
    initialTheta = [initialTheta; W(:)];
end
return