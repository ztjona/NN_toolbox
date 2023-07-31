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
%       *plot_pred       : flag to plot or not the prediction and target of
%                        the network.
%       *plot_pred_freq  : number of epoch to plot progress of the
%                         predictions.
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

    options.plot_pred (1, 1) logical = false;

    options.plot_pred_freq (1,1) double {mustBeInteger, mustBePositive}=10;

    options.plot_freq (1, 1) double {mustBeInteger, mustBePositive} = 10;
    % epochs

    options.verbose_freq (1, 1) double{mustBeInteger, mustBePositive} = 10;
    options.plot_grad (1, 1) logical = false; % very slow
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

%% # Gradient calculation
% -- Backwards propagation
error = cell( 1, obj.n_layers ); % prealloc


%% # Training!
w_flat = flatten( obj.Ws );
bias_mask = flatten( obj.bias_mask ); % as vector

alpha = options.learning_rate;

losses = zeros( 1, options.n_epoch ); % training
losses_test = nan( 1, options.n_epoch ); % testing

% -- initial drawing
if options.plot
    ax = figurePRO(2);
    plot( ax, losses, '.-' );
    plot( ax, losses_test, 'o-' );
    title( ax, "Training progress" );
    xlabel( ax, "Epoch" );
    ylabel( ax, "Loss" );

    legend( ax, "Train", "Test" )
    ax.YScale = 'log';
    ax.Legend.Color = 'none';

    if options.plot_grad
        yyaxis(ax, "right")
        grad_avg = zeros(1, options.n_epoch);
        grad_std = zeros(1, options.n_epoch);
        errorbar(ax, grad_avg, grad_std)
    end
end

class = @(Yi)isequal()
% -- initial drawing of prediction
if options.plot_pred
    switch obj.n_outputs
        case 1
            ax_pred = figurePRO(3);
            scatter( ax_pred, X, Y, 'filled' );
            title( ax_pred, "Evaluation initial" );
            scatter( ax_pred, X_train, obj.propagate( X_train ), '.' );
            scatter( ax_pred, X_test, obj.propagate( X_test ) );
            xlabel( ax_pred, "X" );
            ylabel( ax_pred, "Y" );
            legend( ax_pred, "Data", "Train pred.", "Test pred.", ...
                "Location", "best" )
            ax_pred.Legend.Color = 'none';

            ax_pred.YLim = [-1.1 1.1];
        case 2
            if obj.n_input_feats == 2
            ax_pred = figurePRO(3);
            c = [[1 0 0]; [0 0 1]];
            
            scatter(ax_pred, X(:, 1), X(:, 2),1000, ...
                c(onehotdecode(Y,1:obj.n_outputs, 2), :)*0.8,'filled',MarkerFaceAlpha =0.1,...
            MarkerEdgeAlpha=0.1)

            scatter(ax_pred, X_train(:, 1), X_train(:, 2),10, ...
                c(onehotdecode(Y_train,1:obj.n_outputs, 2), :),"filled", MarkerFaceAlpha =0.7,...
            MarkerEdgeAlpha=0.7)

            scatter(ax_pred, X_test(:, 1), X_test(:, 2),10,...
                c(onehotdecode(Y_test,1:obj.n_outputs, 2), :),MarkerFaceAlpha =0.7,...
            MarkerEdgeAlpha=0.7,Marker="+")

            title( ax_pred, "Evaluation initial" );

            xlabel( ax_pred, "X1" );
            ylabel( ax_pred, "X2" );
            legend( ax_pred, "Data", "Train pred.", "Test pred.", ...
                "Location", "best" )
            ax_pred.Legend.Color = 'none';

            % ax_pred.YLim = [-1.1 1.1];
            else

            end
    end
end
drawnow
% ----------- Loop
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
            error{l} = der_z_i .* obj.d_loss( Y_train, y_pred );
        else
            % without bias
            ws_expanded = reshape( Ws{l + 1}(1:end - 1, :), 1, ...
                n_neurons_after, [] );
            ws_expanded = repmat( ws_expanded, n_observations, 1 );
            back_prop = ws_expanded .* error{l + 1};
            error{l} = der_z_i.* back_prop;
        end

        gradient_unflat{l} = reshape( sum( error{l}.*A_j_L, 1 ), ...
            n_neurons_after, [] )';
    end

    grad = flatten( gradient_unflat );
    grad_avg(i) = mean(grad);
    grad_std(i) = std(grad);
    if options.plot_grad
        yyaxis(ax, "right")
        ax.Children.YData = grad_avg(1:i);
        ax.Children.YNegativeDelta = grad_std(1:i);
        ax.Children.YPositiveDelta = grad_std(1:i);
    end
    % --- SGD!

    % % --- calculate num gradient %% Not working
    % g_num = obj.calculate_numerical_gradient( X_train, Y_train, w_flat );
    %
    % g_diff = sqrt( sum( grad - g_num ).^2 );
    % g_diff

    % ---- update weights
    w_f = w_flat; % previous the update
    w_flat = w_flat - alpha*grad;

    % -- regularization by weight decay
    % bias_mask (boolean vector) avoids regularization of the biases.
    w_flat = w_flat + options.regularization_lambda * (w_f .* bias_mask);

    Ws = unflatten( w_flat );

    % --- plot loss
    if options.plot && mod( i, options.plot_freq ) == 0
        yyaxis(ax, "left")
        ax.Children(2).YData = losses( 1:i );
        ax.XLim = [1 i];
    end

    % -- initial drawing
    if options.plot_pred && mod( i, options.plot_pred_freq ) == 0
        title( ax_pred, sprintf( "Evaluation %d", i ) );
        ax_pred.Children(2).YData = obj.propagate( X_train, Ws );
        ax_pred.Children(1).YData = obj.propagate( X_test, Ws );
    end

    % ---- validation
    if mod( i, options.validation_rate ) == 0
        Ws = unflatten( w_flat ); % updating weights in network

        y_pred_test = obj.propagate( X_test, Ws );

        losses_test(i) = obj.loss_fcn( Y_test, y_pred_test )/n_testing;
        ax.Children(1).YData = losses_test( 1:i );
    end

    % -- output
    if mod( i, options.verbose_freq ) == 0
        fprintf( "%04d\t%.3e\n", i, losses(i) );
    end
    drawnow
end

% ---- unflatten weights
obj.Ws = unflatten( w_f );

end

