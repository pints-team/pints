function [elR0c]=solve_new_param_elasticities_R0(R0,cparam,dR0dc,iplot)
%
%  ***   [elxc,elqc]=solve_new_param_elasticities(t,x,q,cparam,dxdc,dqdc,xdim,kdim,iplot)   ***
%
%  Purpose
%  -------
%  Calculates elasticites of R0 wrt the user-defined parameter
%

elR0c=0;

% Elasticities of R0 wrt user parameter
if abs(R0) > 1.0E-16
    elR0c = cparam/R0*dR0dc;
else
    if(iplot.elasticities == 1 && iplot.cp == 1)
%         fprintf('Elasticity of R0 wrt user parameter: Attempted division by zero \n')
    end
end

end