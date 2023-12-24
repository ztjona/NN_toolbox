function training_plots( obj, switch_plot_type, flag_initial, data )
%training_plots() auxiliary function to plot training progress. The
%required data is passed in the struct variable "data".
%
%
%

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

autor: Jonathan Zea
jonathan.a.zea@ieee.org

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

%}

persistent ax ax_pred % axis of charts, when empty restart chart

%% Loss
switch switch_plot_type
    case "Loss"
        if isempty(ax) || ~isvalid(ax)
            ax = figurePRO(2);
        end

        if flag_initial
            cla(ax)
            plot( ax, data.losses, '.-', DisplayName="Loss" );
            plot( ax, data.losses_test, 'o', DisplayName="Testing" );
            title( ax, "Training progress" );
            xlabel( ax, "Epoch" );
            ylabel( ax, "Loss" );

            ax.YScale = 'log';

            if data.options.plot_include_grad
                yyaxis(ax, "right")
                ylabel( ax, "Gradient" );

                grad_avg = zeros(1, numel( data.losses ) );
                %grad_std = zeros(1, numel( data.losses ) );
                %errorbar( ax, grad_avg, grad_std )
                plot( ax, grad_avg, DisplayName="avg. gradient" )
            end

            legend( ax, "Location", "best" )
            ax.Legend.Color = 'none';
            return;
        end
        % else
        % --- iterations
        yyaxis(ax, "left")
        ax.Children(2).YData = data.losses;
        ax.Children(1).YData = data.losses_test;
        ax.XLim = [1 data.i];


        if data.options.plot_include_grad
            yyaxis(ax, "right")
            %drawnow
            if numel(ax.Children) == 1
                ax.Children.YData = data.grad_avg;
            else
                ax.Children(3).YData = data.grad_avg;
            end
            %ax.Children.YNegativeDelta = grad_std(1:i);
            %ax.Children.YPositiveDelta = grad_std(1:i);
        end

    case "Pred"
        %% Predictions
        if isempty( ax_pred ) || ~isvalid( ax_pred )
            ax_pred = figurePRO(3);
        end

        switch obj.task
            case "regression"
                %% Regression
                if flag_initial
                    cla(ax_pred)
                    scatter( ax_pred, data.X, data.Y, 'filled', ...
                        Tag="Target",DisplayName="Target",Marker="square" );
                    title( ax_pred, "Evaluation initial" );
                    scatter( ax_pred, data.X_train, data.Y_pred_train, ...
                        '.', Tag="train",DisplayName="train" );
                    scatter( ax_pred, data.X_test, data.Y_pred_test ...
                        , Tag="test",DisplayName="test" );

                    xlabel( ax_pred, "X" );
                    ylabel( ax_pred, "Y" );
                    legend( ax_pred, "Location", "best" )
                    ax_pred.Legend.Color = 'none';

                    % ax_pred.YLim = [-1.1 1.1];
                    return;
                end
                % else
                % -- iteration drawing
                title( ax_pred, sprintf( "Evaluation %d", data.i ) );
                n_outputs = size( data.Y_pred_train, 2 );
                for n = 1:n_outputs
                    ax_pred.Children(n_outputs + n).YData = ...
                        data.Y_pred_train(:, n);

                    ax_pred.Children(n).YData = data.Y_pred_test(:, n);
                end




            case "classification"
                %%
                if obj.n_input_feats ~= 2
                    warning("Currently, can only plot in 2D")

                    return;
                end
                %else

                n_outputs = size( data.Y, 2 );
                c = [[1 0 0];[0 1 0]; [0 0 1]];

                if flag_initial
                    cla(ax_pred)

                    scatter(ax_pred, data.X(:, 1), data.X(:, 2), 1000, ...
                        c(onehotdecode(data.Y,1:n_outputs, 2), :)*0.8, ...
                        'filled',MarkerFaceAlpha =0.05,...
                        MarkerEdgeAlpha=0.1, DisplayName = "target")

                    scatter(ax_pred, data.X_train(:, 1), data.X_train(:, 2),20, ...
                        c(onehotdecode(data.Y_pred_train,1:n_outputs, 2), :), ...
                        "filled", MarkerFaceAlpha =0.7,...
                        MarkerEdgeAlpha=0.7, DisplayName="train")

                    scatter(ax_pred, data.X_test(:, 1), data.X_test(:, 2),20,...
                        c(onehotdecode(data.Y_pred_test,1:n_outputs, 2), :), ...
                        MarkerFaceAlpha =0.7,...
                        MarkerEdgeAlpha=0.7,Marker="+", DisplayName = "test")

                    title( ax_pred, "Evaluation initial" );

                    xlabel( ax_pred, "X1" );
                    ylabel( ax_pred, "X2" );
                    legend( ax_pred, "Location", "bestoutside" )
                    ax_pred.Legend.Color = 'none';

                    % ax_pred.YLim = [-1.1 1.1];
                    return;
                end
                %else

                % -- iteration drawing
                title( ax_pred, sprintf( "Prediction at %d", data.i ) );
                ax_pred.Children(1).CData = c(onehotdecode(data.Y_pred_test, 1:n_outputs, 2), :);
                ax_pred.Children(2).CData = c(onehotdecode(data.Y_pred_train, 1:n_outputs, 2), :);

        end
end
drawnow