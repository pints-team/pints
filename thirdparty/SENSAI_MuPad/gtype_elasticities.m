function [elxp]=gtype_elasticities(output_directory,imap,t,x,p,x0,dxdp,...
                                   nt,xdim,kdim,stype,...
                                   iplot,ilist,klist)
%
%   ***  [elxp]=gtype_elasticities(output_directory,imap,t,x,p,x0,dxdp,...
%                                  nt,xdim,kdim,stype,...
%                                  iplot,ilist,klist)  ***
%
%  Purpose
%  -------
%    Computes, prints and plots elasticities of the variables wrt parameters and initial conditions
%
%  Inputs
%  ------
%
%
%
%
%  Calls
%  -----
%    plot_elasticities
%

elxp=zeros(xdim,kdim,nt);
for it=1:nt
    for i=1:xdim
        if abs(x(i,it)) > 1.0E-16
            for k=1:kdim
                elxp(i,k,it)=p(k)/x(i,it)*dxdp(i,k,it);
            end
            if stype == 3
                for j=1:xdim
                    elxp(i,kdim+j,it)=x0(j)/x(i,it)*dxdp(i,kdim+j,it);
                end
            end
        else
            if(iplot.elasticities == 1 && (iplot.dxdp.var == 1 || iplot.dxdp.param == 1))
                fprintf('Elasticity of variables: Attempted division by zero, it=%5i, k=%5i, i=%5i \n', it, k, i)
            end
        end
    end
end


% Prints elasticities of solution with respect to parameters
if (iplot.elasticities == 1 && (iplot.dxdp.true == 1 || iplot.dxdp.var == 1 || iplot.dxdp.param == 1))
    for ii=1:xdim
        for k=1:kdim
            fprintf('Elasticity of x(%3i) wrt parameter %3i at tfinal = %13.6e \n', ii, k, elxp(ii,k,nt))
        end
    end
    fprintf('\n');
end

% Prints elasticities of solution with respect to initial conditions
if stype == 3
    if (iplot.elasticities == 1 && iplot.dxdz)
        for ii=1:xdim
            for k=1:xdim
                fprintf('Elasticity of x(%3i) wrt initial condition %3i at tfinal = %13.6e \n', ii, k, elxp(ii,kdim+k,nt))
            end
        end
        fprintf('\n');
    end
end

% Plot elasticities wrt variables and parameters in lists
if (iplot.elasticities == 1)
    plot_elasticities(output_directory,imap,t,x,elxp,...
                      xdim,kdim,stype,...
                      iplot,ilist,klist)
end

end