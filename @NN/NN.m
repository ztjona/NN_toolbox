classdef NN
    %NN neural network toolbox.
    % The number of layers excludes the input and includes the output.
    % It follows the convention [Weights bias].

    %{
    Laboratorio de Inteligencia y Visión Artificial ESCUELA POLITÉCNICA
    NACIONAL Quito - Ecuador
    
    autor: Jonathan Zea jonathan.a.zea@ieee.org
    
    "I find that I don't understand things unless I try to program them."
    -Donald E. Knuth
    
    13 July 2023
    %}

    %%
    properties (SetAccess=immutable)
        n_input_feats; % number of features in the input
        n_outputs; % number of output neurons
        activations;
        neurons_by_layer; % number of neurons by layer
        n_layers; % number of layers, igonoring input, including outputs
        loss_fcn; % depends on the task: i.e. regression, classification
        n_learnables;% number of learnables
    end
    properties (SetAccess=private)
        Ws; % weights with bias: [Weights' bias']. Note, weights are stored
        % trasposed to save computation time.
    end

    properties (Hidden=true, SetAccess=private)
        bias_mask; %cell with booleans that distinguish the bias from the
        % weights inside the learnable matrix.
        d_loss; % derivative with respect to the prediction y.
    end

    methods
        %% External methods
        % -----------------------------------------------------------------
        obj = train(obj, X, Y, options);

        gradient = calculate_numerical_gradient(obj, X, Y, Ws_flat);

        %% Constructor
        % -----------------------------------------------------------------
        function obj = NN(length_input, neurons_by_layer, task_type, ...
                activation_fncs, options)
            %NN(...)
            %
            %# INPUTS
            %* length_input         -number of features in the input
            %* neurons_by_layer     -array with the number neurons by layer
            %* activation_fncs      -cell with the names of activation
            %                        fncs excluding input and output. By
            %                        default all hidden layers are set with
            %                        relu. It can be a cell with only 1
            %                        activation function, (e.g.
            %                        {"sigmoid"}), in that case, all hidden
            %                        layers will have that function.
            %                        Otherwise the activation_fncs must
            %                        have size equalt to the number of
            %                        layers minus 1. The activation
            %                        function of the last layer is given by
            %                        task_type.
            %* task_type            -"classification" or "regression". It
            %                        can be a member of the enum class
            %                        Loss. This class defines the loss and
            %                        activation functions.
            %  options              -Name value parameters.
            %    * weights_initialization -"glorot": uses glorot algorithm.
            %
            %
            %
            % # Example
            %    >>     NN(3, [2 5 1], "regression", {"tanh", "relu"})
            %

            % # ---- Data Validation
            arguments
                length_input (1, 1) double {mustBePositive, mustBeInteger}
                neurons_by_layer (1, :) double {mustBePositive, mustBeInteger}
                task_type (1,1) Loss
                activation_fncs (1, :) cell = {'relu'};

                options.weights_initialization (1, 1) string = "glorot";
            end

            assert(length(activation_fncs) == 1 || ...
                length(activation_fncs) == numel(neurons_by_layer) - 1, ...
                "%wrong number of activation functions, it is %d, must be %d", ...
                length(activation_fncs), numel(neurons_by_layer) - 1)

            % # ---- preallocs
            obj.neurons_by_layer = neurons_by_layer;
            obj.n_input_feats = length_input;
            obj.n_layers = numel(neurons_by_layer);
            obj.n_outputs = neurons_by_layer(end);
            obj.activations = cell(1, obj.n_layers);

            % # ---- Weights initialization
            % [Weights bias]
            switch options.weights_initialization
                case "glorot"
                    [obj.Ws, obj.n_learnables, obj.bias_mask] = ...
                        NN.glorot_initialization( ...
                        length_input, neurons_by_layer );

                otherwise
                    error( ...
                        "Given algorithm [%s] for weight initialization not defined. Use 'glorot'", ...
                        options.weights_initialization)
            end


            % # ---- activation function
            for i = 1:obj.n_layers - 1

                if numel(activation_fncs) == 1
                    obj.activations{i} = activation_fncs{1};
                else
                    obj.activations{i} = activation_fncs{i};
                end
            end


            % # ---- retrieving from task_type
            obj.activations{end} = task_type.activation_fcn;
            obj.loss_fcn = task_type.loss_fcn;
            obj.d_loss = task_type.d_loss;

        end

        %%
        % -----------------------------------------------------------------
        function obj = replace_weights(obj, Ws_new)
            %obj.update_weights replaces the weights of the network. Keep in
            %mind that biases are concatenated at the last column.
            %
            % # Example
            %>>  net = NN(1, [3 1],"regression");
            %>>  net = net.replace_weights({[W1 b1]', [W2 b2]'})

            % # ---- Data Validation
            arguments
                obj
                Ws_new        (1, :) cell
            end

            % # ----
            for i = 1:obj.n_layers
                assert(isequal(size(Ws_new{i}), size(obj.Ws{i})), ...
                    "new weight size (%d %d) in layer %d", size(Ws_new), i)
            end
            obj.Ws = Ws_new;
        end
        %%
        % -----------------------------------------------------------------
        function [y, As, Zs] = propagate(obj, X, Ws)
            %obj.propagate() runs the forward propagation of the network
            %
            %# INPUTS
            %* X            :n-by-m, n examples, m features
            %* Ws           :weights for the propagation, defaults to the
            %               obj weights.
            %
            %# OUTPUTS
            %* y        :n-by-o, prediction with n examples, o number of
            %           outputs.
            %* As       :cell with every cell corresponding to the input
            %           and the following layers.
            %           It has the values of the after activation values of
            %           each neuron in that layer.
            %* Zs       :cell with every cell corresponding to a layer.
            %           It has the values of the pre activation values of
            %           each neuron in that layer.
            %
            % # Example
            %>>  y = obj.propagate([1 1;2 2;3 3])
            %

            % # ---- Data Validation
            arguments
                obj
                X        (:, :) double
                Ws       (1, :) cell = obj.Ws;
            end

            % # ----
            assert(size(X, 2) == obj.n_input_feats)
            As = cell(1, obj.n_layers + 1);
            Zs = cell(1, obj.n_layers);

            v_ones = ones(size(X, 1), 1);
            As{1} = X;

            for i = 1:obj.n_layers
                Zs{i} =  [As{i} v_ones]*Ws{i};

                As{i+ 1} = NN.apply_activation_fcn(Zs{i}, ...
                    obj.activations{i});
            end
            y = As{end};
        end
    end

    methods (Static)
        %% External static methods
        % -----------------------------------------------------------------
        [Ws, n_learnables, bias_mask] = glorot_initialization( ...
            neurons_by_layer);

        Ws = unflatten_weights(Ws_f, n_inputs, neurons_by_layer);

        ws_f = flat_weights( Ws, n_learnables );

        A_p = apply_deriv_activation_fcn(Z, fcn_name);

        [X_s, mu, sigma] = standarize( X, mu, sigma );

        [Xtrain, Ytrain, Xtest, Ytest] = hold_out( X, Y, train_fraction );
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
                case 'softmax'
                    % In the case of A: 100x3 (100 examples, 3 features)
                    A = exp(Z);
                    A = A./sum(A, 2);
                otherwise
                    error("Activation  function: %s not defined", fcn_name)

            end
        end
    end
end
% More properties at: AbortSet, Abstract, Access, Dependent, GetAccess, ...
% GetObservable, NonCopyable, PartialMatchPriority, SetAccess, ...
% SetObservable, Transient, Framework attributes
% https://www.mathworks.com/help/matlab/matlab_oop/property-attributes.html

% Methods: Abstract, Access, Hidden, Sealed, Framework attributes
% https://www.mathworks.com/help/matlab/matlab_oop/method-attributes.html