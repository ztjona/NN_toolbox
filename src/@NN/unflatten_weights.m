%%
% -------------------------------------------------------------------------
function Ws = unflatten_weights(Ws_f, n_inputs, neurons_by_layer)
%unflatten_weights(...) unflattens the weights
%
% # INPUTS
%  Ws_f                 :1D array flatten representation of weights
%  n_input              :number of features in the input
%  neurons_by_layer     :array with the number neurons by layer
%
% # OUTPUTS
%  Ws                   :cell, with unflatten representation of weights
%
%

% # ---- Data Validation
arguments
    Ws_f    (:, 1)
    n_inputs (1, 1) double {mustBePositive, mustBeInteger}
    neurons_by_layer (1, :) double {mustBePositive, mustBeInteger}
end

% # ----
Ws = cell( 1, length( neurons_by_layer ) );

i1 = 1;
n1 = n_inputs;

for l = 1:length( neurons_by_layer )
    n2 = neurons_by_layer(l);
    Ws{l} = zeros( n1 + 1, n2 );
    i2 = numel( Ws{l} ) + i1;

    Ws{l} = reshape( Ws_f(i1:i2 - 1), [n1 + 1, n2] );

    n1 = n2;
    i1 = i2;
end
end