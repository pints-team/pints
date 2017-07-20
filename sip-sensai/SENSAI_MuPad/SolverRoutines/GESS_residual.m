
function residual = GESS_residual(t,y,x0,p,xdim,kdim,qdim,qFIM)
%  ***   residual = GESS_residual(t,y,x0,p,xdim,kdim,qdim,qFIM)   ***  



residual = 0;
[ntt,tt,xx,dxxdp] = solve_ode(t,x0,p,xdim,kdim,1);

if qFIM==0
    for idim = 1:xdim
        error = y(idim,:)-xx(idim,:);
        residual = residual + error*error';
    end
elseif qFIM == 1
    [qq,dqqdx,dqqdparam]=solve_qoi(t,xx,p,dxxdp,xdim,kdim,qdim)
    for iq = 1:qdim
        error = y(iq,:)-qq(iq,:);
        residual = residual + error*error';
    end
end



end

