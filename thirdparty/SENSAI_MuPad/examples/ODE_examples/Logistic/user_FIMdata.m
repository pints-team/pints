function [eFIM,qFIM,Fdim,Fp,iFtimes,pest,sig]=user_FIMdata

eFIM = 0;
qFIM = 0;

Fdim = 2;
Fp(1) = 1;
Fp(2) = 2;
iFtimes=[0];

pest = 0;
sig = 0.2;

end