function [g] = gvec(t,x,p) 

g1 = -(x(1)/p(2)-1.0)*p(1)*x(1);

g(1) = g1;

end