cc 
rng(7)
%%
nt = 500;
% nt = 50;
n = ceil( nt/0.75 );
[X, Y] = generate_1Ddataset( @(x)sin( x*2*pi ), 1, n );



Xs = NN.standarize( -X );
% figure
% scatter(X, Y,'.')
% hold on
% scatter(Xs, Y,'.')
% legend("org", "standard")


%%
clc
a = NN(1, [3 1], "regression", {'tanh'} )

%%
% Wi = a.Ws;
% a = a.replace_weights( Wi );
%%
rng(7100)
clc
% lr = 0.01/1; %org
lr = 0.001/1;

options = [];
options.validation_rate = 20;
a.train( Xs, Y, ...
    "validation_rate", options.validation_rate, "plot",true...
    ,"plot_include_pred",true, "plot_include_pred_freq", 30 ...
    ,"n_epoch", 100000 ...
    ,learning_rate=lr ...
    ,regularization_lambda=0, ...
    plot_include_grad = true);

%%
rng(7100)
clc
% lr = 0.01/1; %org
lr = 0.0001/1;
options = [];
options.validation_rate = 20;
a.train( Xs, Y, ...
    "validation_rate", options.validation_rate, "plot",true...
    ,"plot_include_pred",true, "plot_include_pred_freq", 40 ...
    ,"n_epoch", 100000 ...
    ,verbose_freq = 20 ...
    , plot_freq=5 ...
    ,learning_rate=lr ...
    ,plot_include_grad=true...
    ,regularization_lambda=0 ...
    ,learning_rate_scheduler="none" ...
) ;
    %,regularization_lambda=0 );


%%
clf(2)
clf(3)
