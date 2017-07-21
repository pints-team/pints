function [elqp]=solve_qoi_elasticities(q,p,x0,dqdparam,nt,xdim,kdim,qdim,stype,iplot)
%
%   ***  [elqp]=solve_qoi_elasticities(q,p,x0,dqdparam,nt,xdim,kdim,qdim,stype,iplot)   ***
%
%  Calculates the elasticities of the QoIs wrt parameters and initial conditions
%
%
%

elqp=zeros(qdim,kdim+xdim,nt);
for it=1:nt
    for iq=1:qdim
        if abs(q(iq,it)) > 1.0E-16
            for k=1:kdim
                elqp(iq,k,it)=p(k)/q(iq,it)*dqdparam(iq,k,it);
            end
            if stype == 3
                for j=1:xdim
                    elqp(iq,kdim+j,it)=x0(j)/q(iq,it)*dqdparam(iq,kdim+j,it);
                end
            end
        else
            if(iplot.elasticities == 1 && iplot.dqdp == 1 | iplot.dqdz == 1)
                fprintf('Elasticity of QoI: Attempted division by zero, it=%5i, iq=%5i,  \n', it, iq)
            end
        end
    end
end


end