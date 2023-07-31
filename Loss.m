classdef Loss
    %Loss

    %{
    Laboratorio de Inteligencia y Visión Artificial
    ESCUELA POLITÉCNICA NACIONAL
    Quito - Ecuador
    
    autor: z_tja
    jonathan.a.zea@ieee.org
    
    "I find that I don't understand things unless I try to program them."
    -Donald E. Knuth
    
    13 July 2023
    %}

    %%
    properties (SetAccess=immutable)
        name;
        loss_fcn; % target | y_pred
        activation_fcn;
        d_loss; % derivative with respect to the prediction y.
        % In keras there is parameter reduction. It can be considered as
        % SUM.
    end

    %%
    enumeration
        
        % half the MSE. for derivative purposes
        % unweighted regression
        % Reduction by sum over batch_size
        regression ( "regression", @(t, y_pred) MSE( t, y_pred )/2, ...
            'purelin', @(t, y_pred) y_pred - t );


        classification ("classification", @ ...
            (t, y_pred) xentropy(t, y_pred), ...
            'softmax', @(t, y_pred) -t./y_pred)
    end

    methods
        %% Constructor
        % -----------------------------------------------------------------
        function obj = Loss(name, loss_fcn, activation_fcn, d_loss)
            %Loss(...) creates a loss function to train neural networks.
            %
            %
            % # INPUTS
            %  name             :name of the loss
            %  loss_fcn         :function handler of the loss function
            %  activation_fcn   :name of the activation function
            %  d_loss           :function handler, derivative of the loss
            %
            % # OUTPUTS
            %  obj
            %
            % # EXAMPLES
            %>>  xe = Loss.regression
            %

            % # ---- Data Validation
            arguments
                name (1, 1) string
                loss_fcn
                activation_fcn (1,:) char
                d_loss
            end

            % # ----
            obj.name = name;
            obj.loss_fcn = loss_fcn;
            obj.activation_fcn = activation_fcn;
            obj.d_loss = d_loss;
        end
    end
end
