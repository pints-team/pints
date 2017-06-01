function [cparam,dcdp] = cp(x,p,xdim,kdim) 

dcdp = zeros(kdim+xdim,1); 

cparam = p(1);

dcdp1 = 1.0;
dcdp2 = 0.0;
dcdp3 = 0.0;

dcdp(1) = dcdp1;
dcdp(2) = dcdp2;
dcdp(3) = dcdp3;

end