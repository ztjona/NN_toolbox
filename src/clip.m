function x = clip(x, x_min, x_max)
%clip() constrains the values of x between x_min and x_max.
%
% # INPUTS
%  x        : value/s to be constrained
%  x_min      : minimum value
%  x_max      : maximum value
%
% # OUTPUTS
%  x        : constrained value/s
%
% # EXAMPLES
%>>     = clip([2 4 5 6, 9, 7], 3, 6)
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: Jonathan Zea
jonathan.a.zea@ieee.org

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

01 August 2023
%}


%%
x(x > x_max) = x_max;
x(x < x_min) = x_min;
