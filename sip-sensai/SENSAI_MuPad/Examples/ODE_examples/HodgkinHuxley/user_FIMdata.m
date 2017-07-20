function [eFIM,qFIM,Fdim,Fp,iFtimes,pest,sig]=user_FIMdata

eFIM = 0;
qFIM = 1;

Fdim = 3;
Fp(1) = 1;
Fp(2) = 2;
Fp(3) = 3;
iFtimes = [];

pest = 0;
sig = 0.2;

end