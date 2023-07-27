function error = MSE(t, y, w)
%MSE() mean square error.
%
% # USAGE
%   error = MSE(t, y);
%
% # INPUTS
%  t        m-by-f, with m examples and f features, target output
%  y        m-by-f, with m examples and f features, predicted output
%  w        1-by-f, vector with weights for the features
%
% # OUTPUTS
%  error    MSE
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: Jonathan Zea
jonathan.a.zea@ieee.org

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

14 July 2023
%}

%% Input Validation
arguments
    t        (:, :)
    y        (:, :)
    w        (1, :) double = 1;
end

assert(isequal(size(t), size(y)), ...
    "target t and prediction y must have the same size")

assert(numel(w) == 1 || size(w, 2) == size(y, 2),"wrong number of weights")

%%
error = mean(mean((t - y).^2, 1).*w, 2); % ?ñ omit nan?
