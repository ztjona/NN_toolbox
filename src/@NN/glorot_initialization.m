function [Ws, n_ws, bias_mask] = glorot_initialization(length_input, ...
    neurons_by_layer)
%glorot_initialization() creates a set of weights and biases using the
%glorot algorithm with normal distribution. Biases are initilized with 0s.
%
% # EQUATIONS:
%
% 1) Var[W_i] = 1/n_i
% The variance of the weights in layer i is equal to the inverse of the
% number of units in that layer
%
% 2) Var[W_i] = 1/n_(i+1)
% The variance of the weights in layer i is equal to the inverse of the
% number of units in the next layer
%
% 3) mean[W_i] = 0
% Follows assumption 3.
%
% 4) For a normal distribution, sigma^2 = 2/(n_i + n_(i+1))
%
%
% # CONSTRAINTS:
%
% 1) Var[A_i] = Var[A_(i+1)]
% The forward propagation is flowing with constant variance
%
% 2) Var[dL/dZ_(i+1)] = Var[dL/dZ_i]
% The backpropagation is flowing with constant variance
%
% # ASSUMPTIONS:
%
% 1) Activation functions are odd, and have unit derivative in 0.
% i.e. f(-x) = -f(x), f'(0) = 1
%
% 2) Inputs and layers are iid (i.e.  independent and identically
% distributed)
%
% 3) Inputs, weights and biases are normalized with 0 means.
%
% # NOTES:
% It does not check that the activation functions follow assumption 1.
%
% # REFERENCES:
%
% 1) https://towardsdatascience.com/xavier-glorot-initialization-in-neural-networks-math-proof-4682bf5c6ec3
%
%
% # USAGE
%   Ws = NN.glorot_initialization(neurons_by_layer);
%
% # INPUTS
%  length_input         :number of features in the input
%  neurons_by_layer     :array with the number of neurons by layer,
%                       excludes the input
%
% # OUTPUTS
%  Ws                   :cell with each element having the weight matrix of
%                       the corresponding layer. Recall, weights and biases
%                       are concatenated as [Weights bias].
%  n_ws                 :number of learnables
%  bias_mask            :cell with booleans that distinguish the bias from
%                       the weights inside the learnable matrix.
%
% # EXAMPLES
%>> NN.glorot_initialization(2, [3 4])
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

15 July 2023
%}

%% Input Validation
arguments
    length_input (1, 1) double {mustBePositive, mustBeInteger}
    neurons_by_layer (1, :) double {mustBePositive, mustBeInteger}
end

%%
n_layers = numel(neurons_by_layer);
Ws = cell(1, n_layers);
bias_mask = cell(1, n_layers);

n0 = length_input;
n_ws = 0;
for i = 1:n_layers
    % [Weights bias]
    n1 = neurons_by_layer(i);
    sigma = sqrt(2/ (n0 + n1));
    Ws{i} = [normrnd(0, sigma, [n0 n1]); zeros( 1, n1 )];

    bias_mask{i} = [false( [n0 n1] ); true( 1, n1 )];

    n_ws = n_ws + numel( Ws{i} );

    n0 = n1;
end










