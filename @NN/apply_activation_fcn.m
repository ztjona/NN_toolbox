%%
% -----------------------------------------------------------------
function A = apply_activation_fcn(Z, fcn_name)
%apply_activation_fcn() applies the given activation function
%to the input Z.
%
%# INPUTS
%  Z
%
%# OUTPUTS
%* A    -> A = fcn_name(Z)
%
% # Example
%>>  A = NN.apply_activation_fcn([.2 .3; .4 .5], "tanh")
%

% # ---- Data Validation
arguments
    Z        (:, :)
    fcn_name (1, 1) string
end

% # ----
switch fcn_name
    case 'tanh'
        A = tanh(Z);
    case 'relu'
        A = Z;
        %A(Z <= 0) = 0;if it is 0, then is already changed to 0
        A(Z < 0) = 0;
    case 'purelin'
        A = Z;
    case 'sigmoid'
        A = 1./(1 + exp( -Z ));
    case 'softmax'
        % In the case of A: 100x3 (100 examples, 3 features)
        A = exp(Z);
        A = A./sum(A, 2);
    otherwise
        error("Activation  function: %s not defined", fcn_name)

end
end