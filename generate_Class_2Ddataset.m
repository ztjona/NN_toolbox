function [X, Y, Y_n] = generate_Class_2Ddataset(n_inputs, dataset_type, ...
    n_points )
%generate_Class_2Ddataset() returns a training dataset for classification with
%given number of points.
%
%
% # INPUTS
%  n_inputs     :number of input features
%  dataset_type    :type of output
%  n_points     :number of sample points [default 100]
%
% # OUTPUTS
%  X            :m-by-f with m examples and f features. Input data.
%  Y            :m-by-o with m examples and o outputs. Target data.
%  Y_n          :m-by-1 with m examples, with classes.
%
% # EXAMPLES
%>>     = generate_Class_2Ddataset()
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

%}

%% Input Validation
arguments
    n_inputs   (1, 1) double {mustBePositive, mustBeInteger}
    dataset_type   (1, 1)
    n_points  (1, 1) double {mustBePositive, mustBeInteger} = 100;
end


%%
X = rand( n_points, n_inputs )*2 - 1;
Y_n = zeros( n_points, 1 );
switch dataset_type
    case "XOR"
        Y_n(X(:, 1) >=0 & X(:, 2) >=0 , :) = 2;
        Y_n(X(:, 1) <0 & X(:, 2) >=0 , :) = 1;
        Y_n(X(:, 1) >=0 & X(:, 2) <0 , :) = 1;
        Y_n(X(:, 1) <0 & X(:, 2) <0 , :) = 2;
    case "Spheres"
        idx1 = X(:, 1).^2 + X(:, 2).^2 <=0.35;
        idx2 = X(:, 1).^2 + X(:, 2).^2 <=0.75;
        Y_n(idx1 & idx2, :) = 1;
        Y_n(~idx1 & idx2, :) = 2;
        Y_n(~idx1 & ~idx2, :) = 3;

end

Y = dummyvar(Y_n);
