function [DIR,JOB,imap,x0,p0,tfinal,solntimes,ntsteps,qtype,stype,NextGen,R0_only]=user_inputs 

DIR = '/home/michael/sensai/HodgkinHuxley';
JOB = 'HH-SpringBreak';

imap = 0;

x0(1,1) = -75;
x0(2,1) = 0.05;
x0(3,1) = 0.6;
x0(4,1) = 0.325;

p0(1,1) = 120;
p0(2,1) = 36;
p0(3,1) = 0.3;

tfinal = 30;
solntimes = [];
ntsteps = 0;

qtype = 0;
stype = 1;

NextGen(1,1) = 0;
R0_only = 0;

end