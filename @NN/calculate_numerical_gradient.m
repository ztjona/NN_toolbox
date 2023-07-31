%%
% -----------------------------------------------------------------
function gradient = calculate_numerical_gradient(obj, X, Y, Ws_flat)
%obj.calculate_numerical_gradient(...) uses the symmetrical finite
%differences method to calculates the gradient.
%
%
% # INPUTS
%  X            :m-by-f with m examples and f features. Input data.
%  Y            :m-by-o with m examples and o outputs. Target data.
%  Ws_flat      :vector of weights to
%
% # OUTPUTS
%  gradient     :vector of the derivate of the loss function with respect
%               of each learnable-
%
%

% # ----

Ws_temp = Ws_flat; % to be updated



gradient = zeros( size( Ws_flat ) );

for i = 1:numel( Ws_flat )
    epsilon = .1;

    % get the gradient of the ith weight
    wi_init = Ws_temp(i);
    

    for c = 1:1000
        Ws_temp(i) = wi_init + epsilon;
        

        wi = NN.unflatten_weights( ...
            Ws_temp, obj.n_input_feats, obj.neurons_by_layer );

        y_p = obj.propagate( X, wi );
        
        Ws_temp(i) = wi_init - epsilon;

        wi = NN.unflatten_weights( ...
            Ws_temp, obj.n_input_feats, obj.neurons_by_layer );

        y_n = obj.propagate( X, wi );

        loss_p = obj.loss_fcn( Y, y_p );
        loss_n = obj.loss_fcn( Y, y_n );

        err = abs( (loss_p - loss_n)  / loss_n );


        if err < 0.001 && err > 0
            break
        elseif err == 0
            epsilon = epsilon*1.1;
        else
            epsilon = epsilon*0.9;
        end
    end

    gradient(i) = err / (2*epsilon);
    Ws_temp(i) = wi_init;
end
end