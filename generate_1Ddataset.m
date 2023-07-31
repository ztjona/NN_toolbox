function [X, Y] = generate_1Ddataset( fcn, n_inputs, n_points )
%generate_1Ddataset() returns a training dataset for regression with given
%function handler, error and number of points.  
%
%
% # INPUTS
%  fcn          :function handler to generate data
%  n_inputs     :number of input features
%  n_points     :number of sample points [default 100]
%
% # OUTPUTS
%  X            :m-by-f with m examples and f features. Input data.
%  Y            :m-by-1 with m examples and 1 output. Target data.
% 
% # EXAMPLES
%>>     = generate_1Ddataset(@(x)sin( x ), 1, n)
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
    fcn        (1, 1) % function handler
    n_inputs   (1, 1) double {mustBePositive, mustBeInteger}
    n_points  (1, 1) double {mustBePositive, mustBeInteger} = 100;
end

assert( isa( fcn, "function_handle" ), ...
    "fcn argument must be function handle" )

%% 
X = rand( n_points, n_inputs );
Y = fcn( X );

