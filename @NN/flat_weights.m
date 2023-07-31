%%
% -------------------------------------------------------------------------
function ws_f = flat_weights( Ws, n_learnables )
%flat_weights(...) flattens the weights into a column vector. 
%
% # INPUTS
%  Ws                   :cell with weights by layer
%  n_learnables         :number of weights and biases
%
% # OUTPUTS
%  ws_f                 :vector, with flatten representation of weights
%
%
%
ws_f = zeros( n_learnables, 1 );

i1 = 1;

for l = 1:length( Ws )
    n1 = size( Ws{l}, 2 );

    i2 = numel( Ws{l} ) + i1;
    ws_f(i1:i2 - 1) = Ws{l}(:);

    i1 = i2;
end
end