function [DIR,JOB,imap,x0,p0,tfinal,solntimes,ntsteps,qtype,stype,NextGen,R0_only]=user_inputs 
%  ***   [DIR,JOB,imap,x0,p0,tfinal,solntimes,ntsteps,qtype,stype,NextGen,R0_only]=user_inputs  ***
%
% Inputs
%  DIR -> Directory in which results are to be stored
%  JOB -> Subdirectory in which results are to be stored
%
%  imap = 0 -> differential equation
%       = 1 -> iterated map
%  x0   = initial condition
%  p0   = parameters
%  tfinal    = final time
%  solntimes = times at which the solution is required
%  ntsteps = 0 -> Matlab solver determines solution times
%                 (computes FIM as a function of time)
%          > 1 -> length of solntimes vector (solntimes(1)=0, the initial time)
%                 (computes FIM based on selected solution times)
%  qtype   = 0 -> non-differentiable function of (x,p)
%          = 1 -> differentiable function of (x,p)
%  stype   = 1 -> solution only
%          = 2 -> solution and sensitivities wrt parameters
%          = 3 -> solution and sensitivities wrt parameters and initial conditions
%
%  NextGen -> input for R0 calculation
%  R0_only -> input for R0 calculation
%

DIR = 'C:/CSU/SENSAI_output/HodgkinHuxley';
JOB = 'HH-4-Jen-May11';

imap = 0;

x0(1,1) = -75;
x0(2,1) = 0.05;
x0(3,1) = 0.4;
x0(4,1) = 0.45;

p0(1,1) = 120;
p0(2,1) = 36;
p0(3,1) = 0.3;

tfinal=40;
solntimes=0;
ntsteps=0;
% solntimes=linspace(0,tfinal,1201);
% ntsteps=length(solntimes);

qtype = 1;
stype = 2;

NextGen(1,1) = 0;
R0_only = 0;

end