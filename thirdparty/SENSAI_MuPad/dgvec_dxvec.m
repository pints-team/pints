function [dgdx] = dgvec_dxvec(t,x,p) 

xdim=length(x);
dgdx = zeros(xdim,xdim); 

dg1dx1 = -(x(1)/p(2)-1.0)*p(1)-(p(1)*x(1))/p(2);

dgdx(1,1) = dg1dx1;

end