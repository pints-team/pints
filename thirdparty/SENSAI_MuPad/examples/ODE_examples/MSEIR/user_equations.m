function [fparam,fvec]=user_equations

xdim = 5;
kdim = 10;
fparam=[xdim,kdim]; 

f1 = 'p[1]*(x[2] + x[3] + x[4] + x[5]) - x[1]*(p[2] + p[3])';
f2 = 'p[2]*x[1] + p[9]*x[5] - x[2]*(p[5] + p[10]) - p[4]*x[2]*x[4]';
f3 = 'p[4]*x[2]*x[4] - (1/p[6] + p[5])*x[3]';
f4 = 'x[3]/p[6] - x[4]*(1/p[8] + p[5] + p[7])';
f5 = 'p[10]*x[2] + 1/p[8] - x[5]*(p[5] + p[9])';

fvec(1,1:length(f1)) = f1;
fvec(2,1:length(f2)) = f2;
fvec(3,1:length(f3)) = f3;
fvec(4,1:length(f4)) = f4;
fvec(5,1:length(f5)) = f5;

end