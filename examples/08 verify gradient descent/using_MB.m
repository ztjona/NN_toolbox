%using_MB implements an example of MB single regression toolbox.
% Hard to compare toolboxes... left behind...

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

23 December 2023
%}

%% Configs


%% Aux and dependent variables
% libs
addpath(genpath('./Toolbox Single Output Multilayer NN Regression/'))


%% Generating dataset
nt = 500;
% nt = 50;
n = ceil( nt/0.75 );
[X, Y] = generate_1Ddataset( @(x)sin( x*2*pi ), 1, n );



Xs = NN.standarize( -X );

%% 
metaParameters.numIterations = 1000;
metaParameters.lambda = 0;
metaParameters.reluThresh = 0;

weights = trainRegressionNN(X, Y, [1 3], {"none", "purelin"}, metaParameters);
