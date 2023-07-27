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
        loss_fcn;
        activation_fcn;
    end

    %%
    enumeration
        
        % half the MSE. for derivative purposes
        regression ("regression", @(t, y_pred) MSE(t, y_pred)/2, "purelin")
        % unweighted regression

        classification ("crossentropy", @(x, y) x, "softmax")
    end

    methods
        %% Constructor
        % -----------------------------------------------------------------
        function obj = Loss(name, loss_fcn, activation_fcn)
            %Loss(...) creates a loss function to train neural networks.
            %
            %
            % # INPUTS
            %  name             :name of the loss
            %  loss_fcn         :function handler of the loss function
            %  activation_fcn   :name of the activation function
            %
            % # OUTPUTS
            %  obj
            %
            % # EXAMPLES
            %>>  xe = Loss.classification
            %

            % # ---- Data Validation
            arguments
                name (1, 1) string
                loss_fcn
                activation_fcn (1,1) string
            end

            % # ----
            obj.name = name;
            obj.loss_fcn = loss_fcn;
            obj.activation_fcn = activation_fcn;
        end
    end
end
