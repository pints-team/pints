function [eFIM,qFIM,Fdim,Fp,pest,sig]=user_FIMdata

eFIM = 0;
qFIM = 0;

Fdim = 4;
Fp(1) = 1;
Fp(2) = 2;
Fp(3) = 3;
Fp(4) = 4;

pest = 0;
sig = 0.2;

end