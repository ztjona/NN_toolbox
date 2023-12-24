function [X_train, Y_train, X_test, Y_test] = hold_out( ...
    X, Y, train_fraction )

%hold_out() splits dataset in train and test subsets.  Does not have data
%balancing. 
%
%
% # INPUTS
%  X            :m-by-f with m examples and f features. Input data.
%  Y            :m-by-o with m examples and o outputs. Target data.
%  train_fraction: number between 0 and 1 indicating the portion of data to
%               be in the training set. 
%
% # OUTPUTS
%  X_train, Y_train, X_test, Y_test
% 
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
    X (:, :) double
    Y (:, :)
    train_fraction = 0.75; % same as Keras
end

assert( size(X, 1) == size(Y, 1), ...
    "Data input and Target must have the same number of observations" )

%% 
n = size( X, 1 );
idxs = randperm( n );
lim = floor( n*train_fraction );

X_train = X(idxs(1:lim), :);
Y_train = Y(idxs(1:lim), :);

X_test = X(idxs(lim + 1:end), :);
Y_test = Y(idxs(lim + 1:end), :);
% 