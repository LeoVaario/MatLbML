function stepConfirm(xdata,w,b,u,c)

%xdata = [0 0 0 0 1 1 1 1; 0 0 1 1 0 0 1 1; 0 1 0 1 0 1 0 1]

%IU = 3;     % a number of input neurons
%HU = 3;     % a number of hidden neurons
%OU = 1;     % a number of output neurons

%w = [  -7.5759    3.6856    3.4257;
%   -5.7869    7.9571   -5.8703;
%    4.8731    5.2036   -7.7549]

%b = [0.2341;
%    1.9020;
 %  -1.1862]

%u = [ 11.6153  -11.8178   11.6295]
%c = [-5.6639]

layer1 = Affine(w,b);
layer2 = Step();
layer3 = Affine(u,c);
layer4 = Step();


%EPOCH=100000; % a number of training epochs
%LAMBDA=0.5; % learning rate

%for epoch=1:EPOCH
  p = layer1.forward(xdata);
  y = layer2.forward(p);
  q = layer3.forward(y);
  z = layer4.forward(q);
  %loss(epoch) = layer5.forward(z,labels);
  
  %calculate gradient
  %dz = layer5.backward();
  %dq = layer4.backward(dz);
  %dy = layer3.backward(dq);
  %dp = layer2.backward(dy);
  %dx = layer1.backward(dp);
  
  %learning weights and biases
  %layer1.update(LAMBDA);
  %layer3.update(LAMBDA);
%end

%loss

% Display loss change graph
%figure(1);
%plot(loss)
%xlabel('Epoch');
%ylabel('LOSS');


z

end
