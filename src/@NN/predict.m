function [Y, varargout] = predict(obj, X)
%predict() returns the prediction of the network. 
%
% # USAGE
%   Y = obj.predict(X) % returns each output neuron prediction by input
%                       example. 
%   [Y, Y_nominal] = obj.predict(X) % returns each output neuron prediction
%                       by input example in the case of classification.
%
% # INPUTS
%  X            :m-by-f with m examples and f features. Input data.
%
% # OUTPUTS
%  Y            :m-by-o with m examples and o outputs. It is the output by
%               neuron.
%  Y_nominal    :m-by-1 with m examples. In the case of classification, the
%               output is a vector of index classes.
% 

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: Jonathan Zea
jonathan.a.zea@ieee.org

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

24 August 2023
%}

%% Input Validation
arguments
    obj (1, 1) NN
    X (:, :) double
end


assert(isequal(size(X, 2), obj.n_input_feats), ...
    "wrong number of features in input matrix X")

%% 
Y = obj.propagate(X);
if nargout == 2 && obj.task == Loss.classification
    [~, varargout{1}] = max(Y,[],2);
end

