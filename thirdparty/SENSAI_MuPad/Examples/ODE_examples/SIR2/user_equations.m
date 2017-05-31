function [fparam,fvec]=user_equations

xdim = 3;
kdim = 4;
fparam=[xdim,kdim]; 

f1 = 'p[4]*x[2] + (x[2] + x[3])*p[2] - p[1]*x[1]*x[2]';
f2 = 'p[1]*x[1]*x[2] - p[3]*x[2] - p[4]*x[2] - p[2]*x[2]';
f3 = 'p[3]*x[2] - p[2]*x[3]';

fvec(1,1:length(f1)) = f1;
fvec(2,1:length(f2)) = f2;
fvec(3,1:length(f3)) = f3;

end