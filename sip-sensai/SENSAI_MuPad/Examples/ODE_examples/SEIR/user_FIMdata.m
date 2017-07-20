function [eFIM,qFIM,Fdim,Fp,pest,sig]=user_FIMdata

eFIM = 1;
qFIM = 1;

Fdim = 4;
Fp(1) = 1;
Fp(2) = 8;
Fp(3) = 3;
Fp(4) = 4;
% Fp(5) = 5;
% Fp(6) = 6;
% Fp(7) = 7;
% Fp(8) = 8;

pest = 0;
sig = 0.2;

end