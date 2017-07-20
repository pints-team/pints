function [fparam,fvec]=user_equations

xdim = 1;
kdim = 2;
fparam=[xdim,kdim]; 

f1 = '-(x[1]/p[2] - 1)*p[1]*x[1]';

fvec(1,1:length(f1)) = f1;

end