function drdp = drhs_dparam(t,x,dxdp,param,xdim,kdim,ncol)
%
%  ***   drdp = drhs_dparam(t,x,dxdp,param,xdim,kdim,ncol)   ***
%
%

drdp=zeros(xdim,ncol);

% Calculate dgdx
dgdx=dgvec_dxvec(t,x,param);

%Calculate dgdparam
dgdp=dgvec_dparam(t,x,param);

% Construct derivative of rhs wrt parameters

for i=1:xdim
    for k=1:ncol
        for m=1:xdim
           drdp(i,k)=drdp(i,k)+dgdx(i,m)*dxdp(m,k);
        end
        drdp(i,k)=drdp(i,k)+dgdp(i,k);
    end
end

end
