function [dgdp] = dgvec_dparam(t,x,p) 

xdim=length(x);
kdim=length(p);
dgdp = zeros(xdim,kdim+xdim); 

dg1dp1 = -(x(1)/p(2)-1.0)*x(1);
dg1dp2 = p(1)*1.0/p(2)^2*x(1)^2;

dgdp(1,1) = dg1dp1;
dgdp(1,2) = dg1dp2;

end