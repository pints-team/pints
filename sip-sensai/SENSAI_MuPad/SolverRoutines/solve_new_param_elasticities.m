function [elxc,elqc]=solve_new_param_elasticities(t,x,q,cparam,dxdc,dqdc,xdim,kdim,iplot)
%
%  ***   [elxc,elqc]=solve_new_param_elasticities(t,x,q,cparam,dxdc,dqdc,xdim,kdim,iplot)   ***
%
%  Purpose
%  -------
%  Calculates elasticities wrt the user-defined parameter
%

nt=length(t);
elxc=zeros(xdim,nt);
elqc=zeros(1,nt);

% Elasticities of the variables and QoI wrt user parameter
for it=1:nt
    for i=1:xdim
        if abs(x(i,it)) > 1.0E-16
               elxc(i,it) = cparam/x(i,it)*dxdc(i,it);
        else
            if(iplot.elasticities == 1 && iplot.cp == 1)
%                 fprintf('Elasticity wrt user parameter: Attempted division by zero, it=%5i, i=%5i \n', it, i)
            end
        end
    end
    if abs(q(it)) > 1.0E-16
         elqc(it) = cparam/q(it)*dqdc(it);
    else
        if(iplot.elasticities == 1 && iplot.cp == 1)
%             fprintf('Elasticity wrt user parameter: Attempted division by zero, it=%5i \n', it)
        end
    end
end

end