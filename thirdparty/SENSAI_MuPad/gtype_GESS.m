function gtype_GESS(DIR,t,y,x0,p,xdim,kdim,qdim,qFIM,pert)
%
%  ***   gtype_GESS(DIR,t,yy,x0,p,xdim,kdim,qdim,qFIM,pert)   ***
%
%  Purpose
%  -------
%  Performs generalized least squares parameter fitting to either
%  variables or quantities of interest
%
%  Calls
%  -----
%    fminsearch
%

global NCOLUMNS

if pert>0
    % Generate data with random normal noise
    [nt,t,x,dxdp]=solve_ode(t,x0,p,xdim,kdim,1);
    y=x+pert*randn(xdim,nt);
    pinit=p.*(1+pert*randn(kdim,1));
else
    % Use input data y
    pinit=p;
end

[pmin,lsqmin] = fminsearch( @(p) GESS_residual(t,y,x0,p,xdim,kdim,qdim,qFIM), pinit);
fprintf('gtype_GESS: Least squares best fit parameters \n')
disp(pmin)
fprintf('gtype_GESS: Least squares error \n')
disp(lsqmin)

end

