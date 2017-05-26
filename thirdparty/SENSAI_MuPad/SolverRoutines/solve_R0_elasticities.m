function [elR0p]=solve_R0_elasticities(p,R0,dR0dp,kdim,iplot)
%
%   ***   function [elR0p]=solve_elasticities(p,x0,R0,dR0dp,xdim,kdim,iplot)   ***
%
%

elR0p=zeros(kdim,1);

% Elasticities of R0 wrt parameters
for k=1:kdim
    if abs(R0) > 1.0E-16
        elR0p(k)=p(k)/R0*dR0dp(k);
    else
%         if(iplot.elasticities == 1 && (iplot.dxdp.var == 1 || iplot.dxdp.param == 1))
%             fprintf('Elasticity of variables: Attempted division by zero for R0, k=%5i, i=%5i \n', k, i)
%         end
    end
end

% % Elasticities of R0 wrt initial conditions
% for j=1:xdim
%     if abs(R0) > 1.0E-16
%         elR0p(kdim+j)=x0(j)/R0*dR0dp(kdim+j);
%     else
% %         if(iplot.elasticities == 1 && iplot.dxdz == 1)
%             fprintf('Elasticity of variables: Attempted division by zero for R0, j=%5i, i=%5i \n', j, i)
% %         end
%     end
% end

end