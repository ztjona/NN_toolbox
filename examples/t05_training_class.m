cc 
rng(7)
%%
nt = 1000;
n = ceil( nt/0.75 );
[X, Y, Y_n] = generate_Class_2Ddataset(2, "XOR", n)


%%
Xs = NN.standarize( X );
X = Xs;
figurePRO
c = [[1 0 0]; [0 0 1]];

scatter(X(:, 1), X(:, 2),10,c(Y_n, :))
hold on
% legend("org", "standard")


%%
clc
a = NN(2, [4 2], "classification", {'tanh'} )

%%
% Wi = a.Ws;
% a = a.replace_weights( Wi );
%%
rng(7100)
clc
lr = 0.001;
options = [];
options.validation_rate = 20;
a.train( Xs, Y, ...
    "validation_rate", options.validation_rate, "plot",true...
    ,"plot_include_pred",true, "plot_include_pred_freq", 40 ...
    ,"n_epoch", 100000 ...
    , plot_freq=5 ...
    , verbose_freq= 40 ...
    , learning_rate= lr ...
    , learning_rate_scheduler="exponential_decay" ...
    ,regularization_lambda=0  ...
    ,plot_include_grad=true); %...
%     );

%%
clf(2)
clf(3)
