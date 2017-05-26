function [eFIM,qFIM,Fdim,Fp,iFtimes,pest,sig]=user_FIMdata
%
% Inputs
% ------
%  eFIM    = 0 -> use sensitivities to construct sensitivity matrix (FIM)
%          = 1 -> use elasticities to construct sensitivity matrix (FIM)
%  Fdim    = number of parameters to construct sensitivity matrix (FIM)
%  Fp      = indices of parameters to construct sensitivity matrix (FIM)
%  iFtimes = indices of times for constructing a global sensitivity matrix (FIM)
%            (for ntsteps > 1)
%
%  pest    = 0 -> no least squares recovery
%          = 1 -> least squares recovery
%  sig     = standard deviation of error to add before least squares recovery
%

eFIM = 0;
qFIM = 1;

Fdim = 3;
Fp(1) = 1;
Fp(2) = 2;
Fp(3) = 3;

%iFtimes=[201;401;601;801;1001;1201];
iFtimes=[0];


pest = 0;
sig = 0.2;

end