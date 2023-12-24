cc all
%%
f1 = @(x)sin(x);
x = linspace(-pi,pi, 50);
y = f1(x);
plot(x, y)
%%
layers = [
    featureInputLayer(1,"Name","featureinput")
    fullyConnectedLayer(4,"Name","fc")
    %reluLayer("Name","relu")
    %
    tanhLayer("Name","tanh")
    fullyConnectedLayer(1,"Name","fc_1")
    regressionLayer("Name","regressionoutput")];

%%
options = trainingOptions("sgdm", ...
    LearnRateSchedule="none", ...
    MaxEpochs=100000, ...
    MiniBatchSize=50, ...
    InitialLearnRate=0.01,...
    Verbose=true,...
    OutputFcn=@(x)fn(x),...
    Plots="training-progress" )
% ...
%     ,...
%     CheckpointPath=".\checks\",...
%     CheckpointFrequency=100,...
%     CheckpointFrequencyUnit="epoch")
%%
net = trainNetwork(x', y',layers,options)

%%
y_pred = net.predict(x');
hold on
plot(x, y_pred, '.:')

%%
legend("org", "pred")