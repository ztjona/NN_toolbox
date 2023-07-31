%%
% -----------------------------------------------------------------
function A = apply_deriv_activation_fcn(Z, fcn_name)
%apply_activation_fcn() applies the derivative of the given activation
%function to the input Z.
%
%# INPUTS
%  Z
%
%# OUTPUTS
%* A    -> A = d fcn_name(Z) / dZ
%
% # Example
%>>  A = NN.apply_deriv_activation_fcn([.2 .3; .4 .5], "tanh")
%

% # ---- Data Validation
arguments
    Z        (:, :)
    fcn_name (1, 1) string
end

% # ----
switch fcn_name
    case 'tanh'
        A = 1 - tanh( Z ).^2;
    case 'relu'
        A = Z;
        A(Z > 0) = 1;
        % A(Z < 0) = 0; % must be already be converted to 0
        % A(Z == 0) = 0; % formally not defined, but...

    case 'purelin'
        A = ones( size ( Z ) );

    case 'softmax'
        % In the case of A: 100x3 (100 examples, 3 features)
        s = NN.apply_activation_fcn( Z, fcn_name ); % could be spared...
        A = s.*(1 - s);


    otherwise
        error("Activation  function: %s not defined", fcn_name)

end
end
