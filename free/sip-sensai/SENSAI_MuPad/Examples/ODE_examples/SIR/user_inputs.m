function [DIR,JOB,imap,x0,p0,tfinal,ntsteps,solution_only,NextGen,R0_only]=user_inputs 

DIR = 'C:/Users/tavener/Dropbox/Research/Current/Mirams/SENSAI_MuPad/Examples/ODE_examples/SIR';
JOB = 'SIR_example';

imap = 0;

x0(1,1) = 600;
x0(2,1) = 0;
x0(3,1) = 0;

p0(1,1) = 0.5;
p0(2,1) = 1000;
p0(3,1) = 0.1;
p0(4,1) = 0.2;
p0(5,1) = 0.02;
p0(6,1) = 0.1;

tfinal = 100;
ntsteps = 0;

solution_only = 0;

NextGen(1,1) = 0;
R0_only = 0;

end