function [eFIM,qFIM,Fdim,Fp,pest,sig]=user_FIMdata

eFIM = 0;
qFIM = 0;

Fdim = 2;
Fp(1) = 1;
Fp(2) = 2;

pest = 1;
sig = 0.2;

end