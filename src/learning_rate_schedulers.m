function lr = learning_rate_schedulers(lr, iter, type)
%learning_rate_schedulers() returns an update of the learning rate, with it
%and the iter as input parameters.
%
% # INPUTS
%  lr       : initial learning rate
%  iter     : index of the iteration
%  type     : string with the name of the scheduler
%
% # OUTPUTS
%  lr       : learning rate
%
% # EXAMPLES
%>>     = learning_rate_schedulers(0.001, 100, "none")
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: z_tja
jonathan.a.zea@ieee.org

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

17 August 2023
%}

%% Input Validation
arguments
    lr (1, 1) double {mustBePositive}
    iter (1, 1) double {mustBePositive, mustBeInteger}
    type (1, 1) string
end

%%
switch type
    case "none"
        return;
    case "exponential_decay"
        k = 1/5000; % in k^-1 iters falls to the 37% of the initial in (2k)^-2 falls to the 13%
        lr = lr*exp(-k*iter);
    otherwise
        error("Learning rate schedule '%s' not defined", type)
end
