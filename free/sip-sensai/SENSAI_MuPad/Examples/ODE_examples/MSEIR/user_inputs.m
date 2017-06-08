function [DIR,JOB,imap,x0,p0,tfinal,ntsteps,solution_only,NextGen,R0_only]=user_inputs 

DIR = 'C:/Users/tavener/Dropbox/Research/Popp/SENSAI_MuPad/Examples/ODE_examples/MSEIR';
JOB = 'MSEIR_example';

imap = 0;

x0(1,1) = 3.25;
x0(2,1) = 270;
x0(3,1) = 0.425;
x0(4,1) = 3.8;
x0(5,1) = 13;

p0(1,1) = 0.000302;
p0(2,1) = 0.16;
p0(3,1) = 0.0045;
p0(4,1) = 0.0034;
p0(5,1) = 0.000252;
p0(6,1) = 0.7;
p0(7,1) = 0.0000175;
p0(8,1) = 0.8;
p0(9,1) = 0.18;
p0(10,1) = 0.026;

tfinal = 100;
ntsteps = 20;

solution_only = 0;

NextGen(1,1) = 0;
R0_only = 0;

end