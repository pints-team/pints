function [ns,msample,vmag,plower,pupper,qlower,qupper,nsim,cvar,nedge]=user_SIP;
%
%   ***  [ns,msample,vmag,plower,pupper,qlower,qupper,nsim,cvar,nedge]=user_SIP;   ***
%  
%  User input for the interface with SIP
%

ns=[1 3 1 1 1  3 1 1 3 3     1 1 1 1 1];
plower=0.8;
pupper=1.2;

% Number of samples in each active subspace 
% Radius of active subspace approximation
msample=2;
vmag=1E-10;

% Bounds on output space
qlower=0.8;
qupper=1.2;

% Number of simulated values
nsim=5000;

% Coefficient of variation of output distribution [SIP]
cvar=0.05;

% Number of bins in each dimension of output space
nedge=11;

end