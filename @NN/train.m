function obj = train(obj, X, Y, options)
%obj.train(...) trains the network with the given data.
%
% # INPUTS
%  X            :m-by-f with m examples and f features. Input data.
%  Y            :m-by-o with m examples and o outputs. Target data.
%  options      :Name value parameters.
%       *learning_rate: initial learning rate.
%       *n_epoch    :number of epoch to train.
%       *validation_rate:number of iterations to reevaluate performance.
%       *batch_size :number of examples to use in the batch size. By
%                    default uses all the dataset.
%       *solver     :solver name.
%
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

    options.learning_rate (1, 1) double {mustBePositive} = 0.001;

    options.n_epoch (1, 1) double {mustBeInteger, mustBePositive} = 1000;

    options.validation_rate (1, 1) double ...
        {mustBeInteger, mustBeNonnegative} = 0;

    options.solver (1, 1) string = "sgd";
end

assert(isequal(size(X, 1), size(Y, 1)), ...
    "number of examples do not match in input X and target Y")

assert(isequal(size(X, 2), obj.n_input_feats), ...
    "wrong number of features in input matrix X")

assert(isequal(size(Y, 2), obj.n_outputs), ...
    "wrong number of outputs Y")

%% # ---- Parameter adaptation
if ~isfield(options, "batch_size")
    options.batch_size = size(X, 1);
else
    % when there is a given batch size
    if options.batch_size > size(X, 1)
        options.batch_size = size(X, 1);
    end
end


% %% tests
% % 3->2-1
% a = {[1 2 3 4; 5 6 7 8]', [1 2 3]'};
% f = flat_weights(a, 11)
%
% ff = unflatten_weights(f, 3,  [2 1])

%% # ---- Calculus of the symbolic gradient
if isempty(obj.loss_sym)


    %--- forward propagation to obtain symbolic weights
    % (Very similar to forward propagation)

    % preallocs
    v_ones = ones( size( X, 1 ), 1 );

    As = cell(1, obj.n_layers + 1);
    % Zs = cell(1, obj.n_layers);

    % sym allocs
    % % (input could not be sym)
    %input_sym = sym( 'in', [size( X, 1 ), obj.n_input_feats] );

    %As{1} = input_sym;
    %input_sym = input_sym(:);

    As{1} =  X;
    Ws_sym = sym(zeros(obj.n_learnables, 1)); % array of symbolic weights

    i = 1; % current index on the weights vector to be updated
    for l = 1:obj.n_layers
        % --- sym agruppation
        % sym weights of the current layer
        ws_sym_l = sym( sprintf( 'w%d_',l ),size( obj.Ws{l} ) );
        i2 = numel(ws_sym_l) + i; % final index + 1
        Ws_sym(i:i2 - 1) = ws_sym_l(:);

        % --- forward propagation
        Z = [As{l} v_ones]*ws_sym_l;
        As{l + 1} = NN.apply_activation_fcn(Z, ...
            obj.activations{l});

        i = i2;
    end

    % % (output can not be sym)
    %t_sym = sym('t', [options.batch_size obj.n_outputs]);
    %t_sym = t_sym(:); % flatteing

    loss_sym = obj.loss_fcn(Y, As{end});

    % --- backing up for possibly next iterations
    obj.loss_sym = loss_sym;
    obj.weights_sym = Ws_sym;

else
    % previously calculated
    loss_sym = obj.loss_sym;
    Ws_sym = obj.weights_sym;
end

% -------- creates symbolic gradient
gradient_sym = sym( zeros( size( Ws_sym ) ) );

for i = 1:length( Ws_sym )
    % with respect to w_i
    gradient_sym(i) = diff( loss_sym, Ws_sym(i) );
end

% ---  creating simpler gradient function handle for new given weights
get_gradient = @(w_i) calculate_gradient( gradient_sym, Ws_sym, w_i );

% --- flattening  weights
flat_w = @(w_i) flat_weights( w_i, length(Ws_sym) ); % even simpler handle
w_f = flat_w( obj.Ws );

uf_w = @(w_i) unflatten_weights( w_i, ...
    obj.n_input_feats, obj.neurons_by_layer ); % even simpler handle

%% # Training!
alpha = options.learning_rate;

for i = 1:options.n_epoch
    fprintf( "%04d\n", i );
    % --- calculate gradient
    grad = get_gradient( w_f );

    % ---- update weights
    w_f = w_f - alpha*grad;

    % ---- validation
    if mod( i, options.validation_rate ) == 0
        obj.Ws = uf_w( w_f ); % updating weights in network
        y_pred = obj.propagate( X );
        loss_i = double( subs( loss_sym, w_f ) );
        fprintf( "%04d\tLoss: %.3e\t\n", i, loss_i );
    end
end

% ---- unflatten weights
obj.Ws = uf_w( w_f );

end

%% ------------------------------------------------------------------------
% Auxs functions
function gradient = calculate_gradient(gradient_sym, Ws_sym, ws_row)
gradient = zeros( size( Ws_sym ) );
for i = 1:length( Ws_sym )
    gradient(i) = double( subs( gradient_sym(i), Ws_sym, ws_row ) );
end
end

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

%%
% -------------------------------------------------------------------------
function ws_f = flat_weights( Ws, n_weights )
% flat_weights
%
ws_f = zeros( n_weights, 1 );
i1 = 1;
for l = 1:length( Ws )
    i2 = numel( Ws{l} ) + i1;
    ws_f(i1:i2 - 1) = Ws{l}(:);
    i1 = i2;
end
end