x = [0,0,1,1;
     0,1,0,1];
t = [0 1 1 0];
 
 
w = [-1,1;
     1,-1];
b = [0.0000000000001;
     0];

u = [1, 1];
c = [0];
     
layer1 = Affine(w,b);
layer2 = Sigmoid();
layer3 = Affine(u,c);
layer4 = Sigmoid();
layer5 = MSE();

p = layer1.forward(x);
y = layer2.forward(p)
q = layer3.forward(y);
z = layer4.forward(q)
loss = layer5.forward(z,t)

% Display a mesh graph of output
figure(2);
[X,Y] = meshgrid(0.00:0.01:1);
xg = [X(:), Y(:)]';
pg = layer1.forward(xg);
yg = layer2.forward(pg);
qg = layer3.forward(yg);
zg = layer4.forward(qg);
surf(X,Y,reshape(double(zg),[101,101]));
