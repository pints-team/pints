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
  dcdp8 = 0.0;
  dcdp9 = 0.0;
  dcdp10 = 0.0;
  dcdp11 = 0.0;
  dcdp12 = 0.0;
  dcdp13 = 0.0;
  dcdp14 = 0.0;
  dcdp15 = 0.0;

dcdp(1) = dcdp1;
dcdp(2) = dcdp2;
dcdp(3) = dcdp3;
dcdp(4) = dcdp4;
dcdp(5) = dcdp5;
dcdp(6) = dcdp6;
dcdp(7) = dcdp7;
dcdp(8) = dcdp8;
dcdp(9) = dcdp9;
dcdp(10) = dcdp10;
dcdp(11) = dcdp11;
dcdp(12) = dcdp12;
dcdp(13) = dcdp13;
dcdp(14) = dcdp14;
dcdp(15) = dcdp15;

end