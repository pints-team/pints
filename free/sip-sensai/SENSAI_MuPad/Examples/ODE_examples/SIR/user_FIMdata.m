function [eFIM,qFIM,Fdim,Fp,pest,sig]=user_FIMdata

eFIM = 0;
qFIM = 0;

Fdim = 6;
Fp(1) = 1;
Fp(2) = 2;
Fp(3) = 3;
Fp(4) = 4;
Fp(5) = 5;
Fp(6) = 6;

pest = 1;
sig = 0.2;

end