function [ngrid,sgrid,msample,vmag,nobs,Qindex,nrsample,obserror,pmin,pmax,qmin,qmax,nsim,cvar,nedge]=user_SIP;
%
%   ***  [ngrid,sgrid,msample,vmag,nrsample,obserror,pmin,pmax,qmin,qmax,nsim,cvar,nedge]=user_SIP;   ***
%  
%  User input for the interface with SIP
%

ngrid=[100 100];
sgrid=0.01;

nobs=1;
Qindex=[1];
nrsample=1;
obserror=0.;

pmin=[0.6; 16];
pmax=[0.8; 19];

% Number of samples in each active subspace 
% Radius of active subspace approximation
msample=0;
vmag=0.01;

% Bounds on output space
qmin=[0.2];
qmax=[90];

% Number of simulated values
nsim=500;

% Coefficient of variation of output distribution [SIP]
cvar=0.01;

% Number of bins in each dimension of output space
nedge=11;

end