cc all
%% sin
[X, Y] = generate_1Ddataset( @(x)sin( x*2*pi ), 1, 1000 )



Xs = NN.standarize( X );
figure
scatter(X, Y,'.')
hold on
scatter(Xs, Y,'.')
legend("prestandar", "standard")

%% Saddle
[X, Y] = generate_1Ddataset( @(x)sin( (x(:, 1) - x(:, 2)).^2 ), 2, 1000 )
scatter(X(:, 1), X(:, 2), 100, Y,'filled')
figure
scatter3(X(:, 1), X(:, 2), Y)