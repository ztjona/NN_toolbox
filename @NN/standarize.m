function [X_s, mu, sigma] = standarize( X, mu, sigma )
%standarize() normalizes the given data using z-score. If mu and sigma
%parameters are not given, those are calculated from data X. 
%
%
% # INPUTS
%  X            :data, columns are features, rows observations. 
%  mu           :mean value
%  sigma        :std
%
% # OUTPUTS
%  X_s          :data standarized
%  mu           :mean value of the data
%  sigma        :std of the data
% 
% # EXAMPLES
%>>    Xs = standarize(rand(100, 2))
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

28 July 2023
%}

%% Input Validation
arguments
    X   (:, :) double
    mu    = [];
    sigma = [];
end
%% 
if isempty( mu )
    mu = mean( X, 1 );
end
if isempty( sigma )
    sigma = std( X, 0, 1 ); % Normalized by N - 1
end

X_s = (X - mu) ./ sigma;

%[X_s, u, s] = zscore(rand(100, 2));

