%check

x = [0,0,1,1;
     0,1,0,1];

w = [2.5,2;
     -2,-2.3];
b = [-0.8;
     3.1];

u = [2, 2.1];
c = [-2.8];
     
layer1 = Affine(w,b);
layer2 = Sigmoid();
layer3 = Affine(u,c);
layer4 = Sigmoid();

p = layer1.forward(x);
y = layer2.forward(p)
q = layer3.forward(y);
z = layer4.forward(q)


% Display a mesh graph of output
figure(1);
[X,Y] = meshgrid(0.00:0.01:1);
xg = [X(:), Y(:)]';
pg = layer1.forward(xg);
yg = layer2.forward(pg);
qg = layer3.forward(yg);
zg = layer4.forward(qg);
surf(X,Y,reshape(double(zg),[101,101]));

