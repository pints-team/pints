function [DIR,JOB,imap,x0,p0,tfinal,solntimes,ntsteps,qtype,stype,NextGen,R0_only]=user_inputs 

DIR = '/home/michael/sensai/logistic';
JOB = 'logistic_example';

imap = 0;

x0(1,1) = 0.1;

p0(1,1) = 0.7;
p0(2,1) = 17.5;

tfinal = 30;
solntimes = 0;
ntsteps = 0;

qtype = 1;
stype = 3;

NextGen(1,1) = 0;
R0_only = 0;

end