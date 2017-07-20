function [cparam,dxdc,dqdc,elxc,elqc] = gtype_cp(output_directory,imap,t,x,p,x0,dxdp,q,dqdparam,...
                                                 nt,xdim,kdim,qdim,iplot,ilist,klist)
%
%   ***  [cparam,dxdc,dqdc,elxc,elqc] = gtype_cp(output_directory,imap,t,x,p,x0,dxdp,q,dqdparam,...
%                                                nt,xdim,kdim,qdim,iplot,ilist,klist)   ***
%
%  Purpose
%  -------
%    Calculates sensitivities and elasticities of variables and quantities of interest
%                wrt a user defined parameter
%
%  Calls
%  -----
%    solve_new_param_sensitivities
%    solve_new_param_elasticities
%


% Calculate sensitivities of variables and QoIs wrt "new" parameter
[cparam,dxdc,dqdc]=solve_new_param_sensitivities(t,x,[p;x0],dxdp,dqdparam,xdim,kdim);

% Print/plot sensitivity of variables and QoIs wrt "new" parameter
if (iplot.sensitivities == 1 && iplot.cp == 1)
    fprintf('\nSensitivity wrt user defined parameter \n')
    for i=1:xdim
        fprintf('Sensitivity of x(%3i) wrt new parameter  at tfinal = %13.6e \n', i, dxdc(i,nt))
    end
    fprintf('Sensitivity of qoi wrt new parameter at tfinal = %13.6e \n\n', dqdc(nt))
      
    % Plot sensitivity of variables and qoi wrt "new" parameter
    plot_new_param_sensitivities(output_directory,imap,t,dxdc,dqdc,xdim,kdim,ilist);
    
end

% Calculate elasticities of variables and QoIs wrt "new" parameter
[elxc,elqc]=solve_new_param_elasticities(t,x,q,cparam,dxdc,dqdc,xdim,kdim,iplot);

% Print elasticities of variables and QoIs wrt user-defined parameter
if (iplot.elasticities == 1 && iplot.cp == 1)
    fprintf('\nElasticity wrt user defined parameter \n')
    for i=1:xdim
        fprintf('Elasticity of x(%3i) wrt new parameter  at tfinal = %13.6e \n', i, elxc(i,nt))
    end
    fprintf('Elasticity of qoi wrt new parameter at tfinal = %13.6e \n\n', elqc(nt))
end


end

