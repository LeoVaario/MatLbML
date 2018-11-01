clear all
 epoch = 1;
epochLimit = 1;
xdata = [0 0 0 0 1 1 1 1; 0 0 1 1 0 0 1 1; 0 1 0 1 0 1 0 1]
%loss = [1]; 
labels = [0 1 1 0 1 0 0 1];

%data_num=4;
 
IU = 3;     % a number of input neurons
HU = 1;     % a number of hidden neurons
%OU = 0;     % a number of output neurons

% initialize weights and biases
% as random numbers between -1.0 and 1.0.
w = 2.0*rand(HU,IU) - 1.0;
b = 2.0*rand(HU,1) - 1.0;
%u = 2.0*rand(OU,HU) - 1.0;
%c = 2.0*rand(OU,1) - 1.0;

layer1 = Affine(w,b);
layer2 = Sigmoid();
%layer3 = Affine(u,c);
%layer4 = Sigmoid();
layer5 = MSE();

%epoch=100000; % a number of training epochs
LAMBDA=0.5; % learning rate

while (epochLimit == 0 || epochLimit == 1000)
    
  p = layer1.forward(xdata);
  y = layer2.forward(p);
  %q = layer3.forward(y);
  %z = layer4.forward(q);
  loss(epochLimit) = layer5.forward(y,labels);
  
  %calculate gradient
  dz = layer5.backward();
  %dq = layer4.backward(dz);
  %dy = layer3.backward(dq);
  dp = layer2.backward(dz);
  dx = layer1.backward(dp);
  
  %learning weights and biases
  layer1.update(LAMBDA);
  %layer3.update(LAMBDA);
  epochLimit = epochLimit +1;
end

loss

% Display loss change graph
figure(1);
plot(loss)
xlabel('Epoch');
ylabel('LOSS');

%stepConfirm(xdata,layer1.weights,layer1.bias,layer3.weights,layer3.bias)
