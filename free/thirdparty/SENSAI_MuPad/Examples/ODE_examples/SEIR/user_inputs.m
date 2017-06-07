function [DIR,JOB,imap,x0,p0,tfinal,ntsteps,solution_only,NextGen,R0_only]=user_inputs 

DIR = 'C:/CSU/SENSAI_MuPad/Examples/ODE_examples/SEIR';
JOB = 'SEIR_example';

imap = 0;

x0(1,1) = 278000;
x0(2,1) = 0.108;
x0(3,1) = 0.189;
x0(4,1) = 721999.703;

p0(1,1) = 1000000;
p0(2,1) = 5;
p0(3,1) = 0.00959;
p0(4,1) = 0.00548;
p0(5,1) = 75;
p0(6,1) = 375;
p0(7,1) = 0.02;
p0(8,1) = -0.02;

tfinal = 6;
ntsteps = 3;

solution_only = 0;

NextGen(1,1) = 0;
R0_only = 0;

end