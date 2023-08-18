function obj = train(obj, X, Y, options)
%obj.train(...) trains the network with the given data.
%
% # INPUTS
%  X            :m-by-f with m examples and f features. Input data.
%  Y            :m-by-o with m examples and o outputs. Target data.
%  options      :Name value parameters.
%       *learning_rate: initial learning rate. DEfault 0.01 (keras, matlab)
%       *n_epoch    :number of epoch to train.
%       *validation_rate:number of iterations to reevaluate performance.
%       *batch_size :number of examples to use in the batch size. By
%                    default uses all the dataset.
%       *solver     :solver name.
%       *regularization_lambda: regularization factor, uses weight decay.
%       When 0 no regularization by weight decay (default 1e-4). The biases
%       are excluded from regularization [Murphy].
%       *hold_out_training: portion between 0 and 1 of observations to use
%                   for gradient calculation. The rest is used as testing
%                   used to report metrics.
%       *plot       : flag to plot or not
%       *plot_freq  : number of epoch to plot progress
%       *plot_include_pred       : flag to plot or not the prediction and
%                               target of the network.
%       *plot_include_pred_freq  : number of epoch to plot progress of the
%                         predictions. Must be multiple of plot_freq.
%
% # OUTPUTS
%  obj          :trained network.
%
%

% # ---- Data Validation
arguments
    obj (1, 1) NN
    X (:, :) double
    Y (:, :)

    options.batch_size (1, 1) double ...
        {mustBeInteger, mustBePositive}; % default use all the examples

    options.learning_rate (1, 1) double {mustBePositive} = 0.01;

    options.n_epoch (1, 1) double {mustBeInteger, mustBePositive} = 1000;

    options.validation_rate (1, 1) double ...
        {mustBeInteger, mustBeNonnegative} = 0;

    options.solver (1, 1) string = "sgd";

    options.regularization_lambda (1, 1) double = 1e-4;

    options.hold_out_training (1, 1) double = 0.75;

    options.plot (1, 1) logical = true;

    options.plot_include_pred (1, 1) logical = false;

    options.plot_include_pred_freq (1,1) double {mustBeInteger, ...
        mustBePositive}=10;

    options.plot_freq (1, 1) double {mustBeInteger, mustBePositive} = 10;
    % epochs

    options.verbose_freq (1, 1) double{mustBeInteger, mustBePositive} = 10;
    options.plot_include_grad (1, 1) logical = false; % very slow
end

assert(isequal(size(X, 1), size(Y, 1)), ...
    "number of examples do not match in input X and target Y")

assert(isequal(size(X, 2), obj.n_input_feats), ...
    "wrong number of features in input matrix X")

assert(isequal(size(Y, 2), obj.n_outputs), ...
    "wrong number of outputs Y")

% %% # ---- Parameter adaptation
% if ~isfield(options, "batch_size")
%     options.batch_size = n_observations;
% else
%     % when there is a given batch size
%     if options.batch_size > n_observations
%         options.batch_size = n_observations;
%     end
% end

%%  Hold out
[X_train, Y_train, X_test, Y_test] = NN.hold_out( ...
    X, Y, options.hold_out_training );

n_observations = size( X_train, 1 );
n_testing = size( X_test, 1 );

%% Auxs functions
% -- Flatten and unflatten
unflatten = @ (Ws_f) NN.unflatten_weights( ...
    Ws_f, obj.n_input_feats, obj.neurons_by_layer );

flatten = @ (Ws) NN.flat_weights( Ws, obj.n_learnables );

error = cell( 1, obj.n_layers ); % prealloc

w_flat = flatten( obj.Ws );
bias_mask = flatten( obj.bias_mask ); % as vector

losses = zeros( 1, options.n_epoch ); % training
losses_test = nan( 1, options.n_epoch ); % testing
grad_avg = zeros( 1, options.n_epoch );

%% Plotting initials
if options.plot
    data.losses = losses;
    data.losses_test = losses_test;
    data.options.plot_include_grad = options.plot_include_grad;

    obj.training_plots( "Loss", true, data );

    if options.plot_include_pred
        data = [];
        data.X = X;
        data.X_train = X_train;
        data.X_test = X_test;
        data.Y = Y;
        data.Y_pred_train = obj.propagate( X_train );
        data.Y_pred_test = obj.propagate( X_test );

        obj.training_plots( "Pred", true, data );
    end
end

%% # Training!
alpha = options.learning_rate;

Ws = unflatten( w_flat );

for i = 1:options.n_epoch
    % --- backpropagate
    gradient_unflat = cell( 1, obj.n_layers ); % same shape as Zs

    % - forward propagation
    [y_pred, As, Zs] = obj.propagate( X_train, Ws );

    % -- loss
    losses(i) = obj.loss_fcn( Y_train, y_pred ) / n_observations;

    % -- backpropagation of the error
    for l = obj.n_layers:-1:1

        n_neurons_after = obj.neurons_by_layer(l); % in the current layer

        % n-observations x n outputs x n inputs (n + bias)
        A_j_L = repmat( ...
            reshape( As{l}, n_observations, 1, [] ),  ...
            1, n_neurons_after );
        % with bias
        A_j_L = cat( 3, A_j_L, ones(n_observations, n_neurons_after ) );


        act_fcn = obj.activations{l};
        der_z_i = NN.apply_deriv_activation_fcn( Zs{l}, act_fcn );

        if l == obj.n_layers
            % last layer - output layer
            % 1-by-n_h_units
            switch obj.task
                case "regression"
                    error{l} = der_z_i .* obj.d_loss( Y_train, y_pred );
                case "classification"
                    error{l} = obj.d_loss( Y_train, y_pred );
            end
        else
            % without bias
            ws_expanded = reshape( Ws{l + 1}(1:end - 1, :), 1, ...
                n_neurons_after, [] );
            ws_expanded = repmat( ws_expanded, n_observations, 1 );

            err_expanded = repmat( reshape( ...
                error{l + 1}, n_observations, 1, [] ), 1, n_neurons_after);

            back_prop = ws_expanded .* err_expanded;
            error{l} = sum( der_z_i.* back_prop, 3 );
        end

        gradient_unflat{l} = reshape( sum( error{l}.*A_j_L, 1 ), ...
            n_neurons_after, [] )';
    end

    grad = flatten( gradient_unflat );
    grad_avg(i) = mean( abs( grad ) );

    % --- SGD!

    % % --- calculate num gradient for comprobation! 
    % Working!
    %g_num = obj.calculate_numerical_gradient( X_train, Y_train, w_flat );
    %
    % must be very low
    %g_diff = sqrt( mean( (grad - g_num ).^2 ) )
    
    % ---- update weights
    w_f = w_flat; % previous the update
    w_flat = w_flat - alpha*grad;

    % -- regularization by weight decay
    % bias_mask (boolean vector) avoids regularization of the biases.
    w_flat = w_flat + options.regularization_lambda * (w_f .* bias_mask);

    Ws = unflatten( w_flat );

    % --- plot loss
    if options.plot && mod( i, options.plot_freq ) == 0
        data.i = i;
        data.losses = losses(1:i);

        data.options.plot_include_grad = options.plot_include_grad;

        if options.plot_include_grad
            data.grad_avg = grad_avg(1:i);
        end



        % - validation
        if mod( i, options.validation_rate ) == 0
            y_pred_test = obj.propagate( X_test, Ws );
            losses_test(i) = obj.loss_fcn( Y_test, y_pred_test )/n_testing;
        end

        data.losses_test = losses_test(1:i);

        obj.training_plots( "Loss", false, data );
    end

    % -- plot pred
    if options.plot_include_pred && ...
            mod( i, options.plot_include_pred_freq ) == 0
        data.i = i;

        data.Y_pred_train = obj.propagate( X_train, Ws );
        data.Y_pred_test = obj.propagate( X_test, Ws );

        obj.training_plots( "Pred", false, data );
    end



    % -- output
    if mod( i, options.verbose_freq ) == 0
        fprintf( "%04d\t%.3e\n", i, losses(i) );
    end
    % --
    if options.plot

    end
    drawnow
end

% ---- unflatten weights
obj.Ws = unflatten( w_f );

end

