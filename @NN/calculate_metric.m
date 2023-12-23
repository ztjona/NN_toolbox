function score = calculate_metric(Y_target, Y_pred, metric)
%NN.calculate_metric() calculates the desired metric.
%
%
% # INPUTS
%  Y_target     :m-by-o with m examples and o outputs. Target data.
%  Y_pred       :m-by-o with m examples and o outputs. Predicted data.
%  metric       :name of the metric. Options are:
%               "accuracy"  :percentage of correctly predicted over total
%
% # OUTPUTS
%  score        :result of the metric applied on the y pred and target.
%
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
    Y_target    (:, :) double
    Y_pred      (:, :) double
    metric      (1, 1) string
end

assert(isequal(size(Y_target), size(Y_pred)), ...
    "Target and prediction does not have the same size")

%%
switch metric
    case "accuracy"
        % categorizing
        if size(Y_target, 2) > 1
            Y_target = NN.categorize(Y_target);
        end

        if size(Y_pred, 2) > 1
            Y_pred = NN.categorize(Y_pred);
        end

        score = mean(Y_target == Y_pred)*100;
    otherwise
        error("Metric '%s' not defined", metric)
end