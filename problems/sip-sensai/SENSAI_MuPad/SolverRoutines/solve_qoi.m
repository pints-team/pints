function [q,dqdx,dqdparam]=solve_qoi(t,x,p,dxdp,xdim,kdim,qdim,stype)
%
%  ***   [q,dqdx,dqdparam]=solve_qoi(t,x,p,dxdp,xdim,kdim,qdim,stype)   ***
%
%  Calculates the QoIs and their derivatives wrt the variables, parameters 
%  and initial conditions
%
%  Calls
%  -----
%    qoi
%
%

[nx,nt]=size(x);
if nt ~= length(t) 
    fprintf('qoi:  Houston, we have a problem in x \n')
end
np=length(p);
if np ~= kdim+xdim
    fprintf('qoi:  Houston, we have a problem in p \n')
end

% Compute quantity of interest
q=zeros(qdim,nt);
dqdx=zeros(qdim,xdim,nt);
dqdp=zeros(qdim,kdim+xdim,nt);
for it=1:nt
    [q(1:qdim,it),dqdx(1:qdim,1:xdim,it),dqdp(1:qdim,1:kdim+xdim,it)]...
          =qoi(t(it),x(1:xdim,it),p,xdim,kdim,qdim);
end

% Compute sensitivity of quantity of interest wrt parameters
if stype == 2
    ncol=kdim;
elseif stype == 3
    ncol=kdim+xdim;
end

dqdparam=zeros(qdim,1,nt);
if stype >= 2
    dqdparam=zeros(qdim,ncol,nt);
    for it=1:nt
        for iq=1:qdim
            for k=1:ncol
                dqdparam(iq,k,it)=0;
                for m=1:xdim
                    dqdparam(iq,k,it)=dqdparam(iq,k,it) + dqdx(iq,m,it)*dxdp(m,k,it);
                end
                dqdparam(iq,k,it)=dqdparam(iq,k,it) + dqdp(iq,k,it);
            end
        end
    end
end

end
