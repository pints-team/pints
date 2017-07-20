function [g]=gsystem(t,y,p,xdim,kdim,ncol)
%
%  ***   [g]=gsystem(t,y,p,xdim,kdim,ncol)   ***
%
%  Input
%     t = time
%     y = current solution of length xdim*(1+kdim+xdim) (=ny)
%     p = vector of parameters of length kdim
%     xdim = number of variables
%     kdim = number of parameters
%     ncol = number of "parameters" for which sensitivies are sought
%
%  Output
%      g = rhs vector for the system of odes of length xdim*(1+kdim+xdim)

% Establish g and dgdp
ny=length(y);
g=zeros(ny,1);

dgdp=zeros(xdim,ncol);

% Establish x and dxdp
x=zeros(xdim,1);
dxdp=zeros(xdim,ncol);

% Unpack x vector from y vector
x=y(1:xdim);

% Unpack dxdp matrix from y vector
if ncol > 0
    dxdp=reshape(y(xdim+1:ny), xdim, ncol);
end


% Calculate g
g(1:xdim)=gvec(t,x,p);

% Calculate dgdp
if ncol > 0
    dgdp=drhs_dparam(t,x,dxdp,p,xdim,kdim,ncol);
    % Repack dgdp as a column vector and add to g
    % (In fact this is unnecessary as it is the default, but it makes the point)
    g(xdim+1:ny)=reshape(dgdp,xdim*ncol,1);
end

end