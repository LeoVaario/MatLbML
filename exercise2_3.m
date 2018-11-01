x = [0,0,1,1;
     0,1,0,1];

t = [0 1 1 0];
 
w = [-1,1;
     1,-1];
b = [0;
     0];

u = [1, 1];
c = [0];
     
layer1 = Affine(w,b);
layer2 = Step();
layer3 = Affine(u,c);
layer4 = Step();
layer5 = MSE();

p = layer1.forward(x);
y = layer2.forward(p)
q = layer3.forward(y);
z = layer4.forward(q)
loss = layer5.forward(z,t)
