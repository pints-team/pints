function solution_sensitivity_output(output_directory,imap,t,x,dxdp,...
                                     nt,xdim,kdim,iplot,ilist,klist,...
                                     solution_only)
%
%   ***  solution_sensitivity_output(output_directory,imap,t,x,dxdp,...
%                                    nt,xdim,kdim,iplot,ilist,klist,...
%                                    solution_only)   ***
%
%  Prints and plots solutions and sensitivities of the solution
%  variables wrt parameters and initial conditions
%
%  Calls
%  -----
%    plot_solutions
%    plot_warnings
%    plot_sensititives
%

% Prints the solution at tfinal (Always printed)
fprintf('\n');
for ii=1:xdim
    fprintf('x(%3i) at tfinal = %20.13e \n', ii, x(ii,nt))
end
fprintf('\n');

% Plot solution
plot_solutions(output_directory,imap,t,x,ilist)

% Plot warnings
plot_warnings(iplot,solution_only)

if(solution_only ~= 0)
    
    % Prints sensitivity of solution with respect to parameters
    if (iplot.sensitivities == 1 && (iplot.dxdp.true == 1 || iplot.dxdp.var == 1 || iplot.dxdp.param == 1))
        for ii=1:xdim
            for k=1:kdim
                fprintf('Sensitivity of x(%3i) wrt parameter %3i at tfinal = %13.6e \n', ii, k, dxdp(ii,k,nt))
            end
        end
        fprintf('\n');
    end
    
    % Prints sensitivity of solution with respect to initial conditions
    if (iplot.sensitivities == 1 && iplot.dxdz)
        for ii=1:xdim
            for k=1:xdim
                fprintf('Sensitivity of x(%3i) wrt initial condition %3i at tfinal = %13.6e \n',...
                    ii, k, dxdp(ii,kdim+k,nt))
            end
        end
        fprintf('\n');
    end
    
    % Plot sensitivities wrt variables and parameters in lists
    if (iplot.sensitivities == 1)
        plot_sensitivities(output_directory,imap,t,x,dxdp,xdim,kdim,iplot,ilist,klist)
    end
    
end

    
end

