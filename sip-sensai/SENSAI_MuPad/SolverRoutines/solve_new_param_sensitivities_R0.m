function [dR0dc]=solve_new_param_sensitivities_R0(x0,p,dR0dp,xdim,kdim)
%
%  ***   [dR0dc]=solve_new_param_sensitivities_R0(x0,p,dR0dp,xdim,kdim)   ***
%
%
%  Purpose
%  -------
%  Calculates sensitivities of R0 wrt the user-defined parameter
%

dR0dc=0;
[cparam,denom]=cp(x0,p,xdim,kdim);

for k=1:kdim
    if denom(k) ~= 0
        dR0dc = dR0dc + dR0dp(k)  / denom(k);
    end
end

end
 
