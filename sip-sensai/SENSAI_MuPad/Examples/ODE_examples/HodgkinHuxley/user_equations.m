function [fparam,fvec]=user_equations

xdim = 4;
kdim = 3;
fparam=[xdim,kdim]; 

f1 = '- (x[1] - 40)*p[1]*x[3]*x[2]^3 - (x[1] + 87)*p[2]*x[4]^4 - (x[1] + 64.387)*p[3] + 20';
f2 = '((x[2] - 1)*(0.1*x[1] + 5.0))/(exp(- x[1]/10 - 5) - 1) - 4*exp(- x[1]/18 - 25/6)*x[2]';
f3 = '- 0.07*exp(- x[1]/20 - 15/4)*(x[3] - 1) - x[3]/(exp(- x[1]/10 - 9/2) + 1)';
f4 = '((x[4] - 1)*(0.01*x[1] + 0.65))/(exp(- x[1]/10 - 13/2) - 1) - 0.125*exp(- x[1]/80 - 15/16)*x[4]';

fvec(1,1:length(f1)) = f1;
fvec(2,1:length(f2)) = f2;
fvec(3,1:length(f3)) = f3;
fvec(4,1:length(f4)) = f4;

end