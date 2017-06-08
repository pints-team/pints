function [q,dqdx,dqdparam,elqp] = gtype_qoi(output_directory,imap,t,x,p,x0,dxdp, ...
                                            nt,xdim,kdim,qdim,stype,...
                                            iplot,ilist,klist)
%
%   ***  [q,dqdx,dqdparam,elqp] = gtype_qoi(output_directory,imap,t,x,p,x0,dxdp,...
%                                           nt,xdim,kdim,qdim,stype,
%                                           iplot,ilist,klist)   ***
%
%  Purpose
%  -------
%    Calculates, prints and plots the quantities of interest and their sensitivities and elasticities
%    wrt to parameters and initial conditions
%
%  Calls
%  -----
%    solve_qoi
%    solve_qoi_elasticities
%    plot_qoi_sensitivities
%    plot_qoi_elasticities
%

% Calculate quantity of interest
[q,dqdx,dqdparam]=solve_qoi(t,x,[p;x0],dxdp,xdim,kdim,qdim,stype);

% Print quantity of interest sensitivities wrt parameters at tfinal
if (iplot.sensitivities == 1 && iplot.dqdp == 1)
    for iq=1:qdim
        fprintf('qoi(%2i) at tfinal = %13.6e \n', iq, q(iq,nt))
        for k=1:kdim
            fprintf('Sensitivity of qoi(%2i) wrt parameter %3i at tfinal = %13.6e \n',...
                iq, k, dqdparam(iq,k,nt))
        end
        fprintf('\n');
    end
end

% Print quantity of interest sensitivities wrt initial conditions at tfinal
if stype == 3
    if (iplot.sensitivities == 1 && iplot.dqdz == 1)
        for iq=1:qdim
            fprintf('qoi(%2i) at tfinal = %13.6e \n', iq, q(iq,nt))
            for k=1:xdim
                fprintf('Sensitivity of qoi(%2i) wrt initial condition %3i at tfinal = %13.6e \n',...
                    iq, k, dqdparam(iq,kdim+k,nt))
            end
            fprintf('\n');
        end
    end
end

if stype == 2
    ncol=kdim;
elseif stype == 3
    ncol=kdim+xdim;
end

% Plot sensitivities wrt qoi
if (iplot.sensitivities == 1)
    for iq=1:qdim
        dqdp_plot(1:ncol,1:nt)=dqdparam(iq,1:ncol,1:nt);
        plot_qoi_sensitivities(output_directory,imap,iq,t,q(iq,1:nt),dqdp_plot,...
            xdim,kdim,stype,...
            iplot,ilist,klist)
    end
end

[elqp]=solve_qoi_elasticities(q,p,x0,dqdparam,nt,xdim,kdim,qdim,stype,iplot);

% Print QoI elasticity wrt parameters
if(iplot.elasticities == 1 && iplot.dqdp == 1)
    for iq=1:qdim
        for k=1:kdim
            fprintf('Elasticity of qoi(%3i) to parameter %3i at tfinal = %13.6e \n',...
                iq, k, elqp(iq,k,nt))
        end
    end
    fprintf('\n');
end

% Print QoI elasticity wrt initial conditions
if stype == 3
    if(iplot.elasticities == 1 && iplot.dqdz == 1)
        for iq=1:qdim
            for k=1:xdim
                fprintf('Elasticity of qoi(%3i) to initial condition %3i at tfinal = %13.6e \n',...
                    iq, k, elqp(iq,k+kdim,nt))
            end
        end
        fprintf('\n');
    end
end




% Plot elasticities of QoI wrt parameters
if (iplot.elasticities == 1)
    for iq=1:qdim
        elqp_plot(1:ncol,1:nt)=elqp(iq,1:ncol,1:nt);
        plot_qoi_elasticities(output_directory,imap,iq,t,q(iq,1:nt),elqp_plot,...
                              xdim,kdim,stype,...
                              iplot,ilist,klist)
    end
end



end

