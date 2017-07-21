function [fparam,fvec]=user_equations

xdim = 4;
kdim = 8;
fparam=[xdim,kdim]; 

f1 = 'p[1]/p[5] + x[4]/p[2] - x[1]/p[5] - (p[6]*x[1]*x[3]*(cos(2*pi*t)*p[7] + sin(2*pi*t)*p[8] + 1))/p[1]';
f2 = '(p[6]*x[1]*x[3]*(cos(2*pi*t)*p[7] + sin(2*pi*t)*p[8] + 1))/p[1] - x[2]/p[5] - x[2]/p[4]';
f3 = 'x[2]/p[4] - x[3]/p[3] - x[3]/p[5]';
f4 = 'x[3]/p[3] - x[4]/p[2] - x[4]/p[5]';

fvec(1,1:length(f1)) = f1;
fvec(2,1:length(f2)) = f2;
fvec(3,1:length(f3)) = f3;
fvec(4,1:length(f4)) = f4;

end