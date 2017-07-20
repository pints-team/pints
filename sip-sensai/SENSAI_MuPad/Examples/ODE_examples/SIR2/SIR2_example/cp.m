function [cparam,dcdp] = cp(x,p,xdim,kdim) 

dcdp = zeros(kdim+xdim,1); 

  cparam = p(1);

  dcdp1 = 1.0;
  dcdp2 = 0.0;
  dcdp3 = 0.0;
  dcdp4 = 0.0;
  dcdp5 = 0.0;
  dcdp6 = 0.0;
  dcdp7 = 0.0;

dcdp(1) = dcdp1;
dcdp(2) = dcdp2;
dcdp(3) = dcdp3;
dcdp(4) = dcdp4;
dcdp(5) = dcdp5;
dcdp(6) = dcdp6;
dcdp(7) = dcdp7;

end