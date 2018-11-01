%Formal Neuron
function x = fn(x, w, h)
    p = w*x;
    x = p>h;
end