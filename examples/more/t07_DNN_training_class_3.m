cc 
rng(7)
%%
nt = 10000;
n = ceil( nt/0.75 );
[X, Y, Y_n] = generate_Class_2Ddataset(2, "Spheres", n)


%%
Xs = NN.standarize( X );
X = Xs;
figurePRO
c = [[1 0 0]; [0 1 0]; [0 0 1]];

scatter(X(:, 1), X(:, 2),10,c(Y_n, :))
hold on
% legend("org", "standard")


%%
% clc
% a = NN(2, [5 3], "classification", {'tanh'} )

%%  Hold out
[X_train, Y_train, X_test, Y_test] = NN.hold_out( ...
    X, Y, 0.75 );

Y_test_cat = NN.categorize(Y_test);
Y_train_cat = NN.categorize(Y_train);

%%
layers = [
    featureInputLayer(2,"Name","featureinput")
    fullyConnectedLayer(5,"Name","fc_1")
    tanhLayer("Name","tanh")
    fullyConnectedLayer(3,"Name","fc_2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

%%
options = trainingOptions("sgdm", ...
    LearnRateSchedule="none", ...
    MaxEpochs=100000, ...
    MiniBatchSize=5000, ... % very big
    InitialLearnRate=1,...
    L2Regularization=0,...
    Verbose=true,...
    ValidationData={X_test, categorical(Y_test_cat)}, ...
    Momentum = 0,...
    Shuffle="never",...
     ...
    ...%OutputFcn=@(x)fn(x),...
    Plots="training-progress" )
% ...
%     ,...
%     CheckpointPath=".\checks\",...
%     CheckpointFrequency=100,...
%     CheckpointFrequencyUnit="epoch")
%%
net = trainNetwork(X_train, categorical(Y_train_cat), layers,options)