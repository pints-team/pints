function [cparam,dxdc,dqdc]=solve_new_param_sensitivities(t,x,p,dxdp,dqdparam,xdim,kdim)
%
%  ***   [cparam,dxdc,dqdc]=solve_new_param_sensitivities(t,x,p,dxdp,dqdparam,xdim,kdim)   ***
%
%  Purpose
%  -------
%  Calculates sensitivities wrt the user-defined parameter
%

nt=length(t);

dxdc=zeros(xdim,nt);
dqdc=zeros(1,nt);
for it=1:nt
    [cparam,denom]=cp(x(:,it),p,xdim,kdim);
    tmp1=zeros(xdim,1);
    tmp2=0;
    for k=1:kdim+xdim
        if denom(k) ~= 0
            tmp1 = tmp1 + dxdp(1:xdim, k, it) / denom(k);
            tmp2 = tmp2 + dqdparam(k,it)  / denom(k);
        end
    end
    dxdc(1:xdim, it)=tmp1;
    dqdc(it)=tmp2;
end