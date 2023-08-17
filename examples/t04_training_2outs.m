cc 
rng(7)
%%
nt = 50;
n = ceil( nt/0.75 );
[X, Y] = generate_1Ddataset( @(x)[sin( x*2*pi ) -sin( x*2*pi )/10], 1, n );



Xs = NN.standarize( -X );
figure
scatter(X, Y,'.')
hold on
scatter(Xs, Y,'.')
legend("org", "standard")


%%
clc
a = NN(1, [3 2], "regression", {'tanh'} )

%%
% Wi = a.Ws;
% a = a.replace_weights( Wi );
%%
rng(7100)
clc
lr = 0.01/1*0.5;
options = [];
options.validation_rate = 20;
a.train( Xs, Y, ...
    "validation_rate", options.validation_rate, "plot",true...
    ,"plot_include_pred",true, "plot_include_pred_freq", 10 ...
    ,"n_epoch", 100000 ...
    ,learning_rate=lr ...
    ,plot_include_grad=true...
    ,regularization_lambda=0 );

%%
clf(2)
clf(3)
