function score = evaluate(obj, X, Y_target, Y_pred, metric)
%evaluate() returns the score obtained with the given metric
%
% # USAGE
%   [] = evaluate();
%
% # INPUTS
%  X            :m-by-f with m examples and f features. Input data.
%  Y_target     :m-by-o with m examples and o outputs. Target data.
%  metric       :name of the metric
%
% # OUTPUTS
%  score        :result of the metric applied on the y pred and target.
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
    Y_target (:, :) double
    Y_pred (:, :) double = [];
    metric string = "accuracy";
end

%%
if isempty(Y_pred)
    Y_pred = obj.predict(X);
end

score = NN.calculate_metric(Y_target, Y_pred, metric);

