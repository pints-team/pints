function [ngrid,sgrid,msample,vmag,nobs,Qindex,nrsample,obserror,pmin,pmax,qmin,qmax,nsim,cvar,nedge]=user_SIP;
%   ***  [ngrid,sgrid,msample,vmag,Qindex,nrsample,obserror,pmin,pmax,qmin,qmax,nsim,cvar,nedge]=user_SIP;   ***
%  
% Inputs
% ------
%  ngrid = number of volumes in each parameter 
%  sgrid = grid stretching factor
%  nobs  = ??? (may be redundant)
%  Qindex = Quantities of interest to be computed
%  nrsample = number of random samples
%  obserror = assumed observation error
%  pmin = minimum values of parameters
%  pmax = maximum values of parameters
%
%  msample = number of samples in local linear approximation based on FIM
%  vmag = maximum distance along singular vectors
%  qmin = ??? (may be redundant)
%  qmax = ??? (may be redundant) 
%  nsim = number of simulated values
%  cvar  = ???  (may be redundant) 
%  nedge = ??? (may be redundant) 

ngrid=[10 10 10];
sgrid=0.01;

nobs=1;
Qindex=[1 2 3 4 5];
nrsample=1;
obserror=0.0;

pmin=[60;  18; 0.15];
pmax=[240; 72; 0.60];

% Number of samples in each active subspace 
% Radius of active subspace approximation
msample=0;
vmag=0.05;

% Bounds on output space
qmin=[2.0; 28];
qmax=[2.4; 34];

% Number of simulated values
nsim=500;

% Coefficient of variation of output distribution [SIP]
cvar=0.05;

% Number of bins in each dimension of output space
nedge=11;

end