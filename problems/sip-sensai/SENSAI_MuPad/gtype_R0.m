function [r0,dR0dp,R0warnings] = gtype_R0(output_directory,imap,p,x,dxdp,cparam,...
                                          nt,xdim,kdim,NextGen,R0_only,...
                                          iplot,ilist,klist)
%
%    *** [r0,dR0dp,R0warnings] = gtype_R0(output_directory,imap,p,x,dxdp,cparam,...
%                                         nt,xdim,kdim,NextGen,R0_only,...
%                                         iplot,ilist,klist)
%  Purpose
%  --------
%    Calculates, prints and plots R0, its sensitivities and elasticities
%
%  Calls
%  -----
%    solve_R0
%    R0_hypothesis_check
%    solve_R0_elasticities
%    solve_new_param_sensitivities_R0
%    plot_R0
%

% Compute R0 and its sensitivities
DFx0 = x(:,1);     % create a disease-free initial condition
for ii=NextGen
    DFx0(ii)= 0;
end

% Compute R0 and its sensitivities
[r0,dR0dp,R0warnings] = solve_R0(DFx0,p,dxdp(:,1:kdim,end),kdim,NextGen);   % Not quite correct

% See if there is a condition in the theorem that does not hold!
R0_hypothesis_check(r0,dR0dp,R0warnings)

% Print R0 and its sensitivities
fprintf('\nR0 = %5.4d \n', r0)
if R0_only == 0
    for k=1:kdim
        fprintf('Sensitivity of R0 wrt parameter %3i = %13.6e \n', k, dR0dp(k))
    end
    fprintf('\n');
end

% Compute R0 elasticities
[elR0p]=solve_R0_elasticities(p,r0,dR0dp,kdim,iplot);
if R0_only == 0
    for k=1:kdim
        fprintf('Elasticity of R0 to parameter %3i at t=0 is %13.6e\n', k, elR0p(k))
    end
    fprintf('\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                 User specified parameter                                %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Sensitivity of R0 wrt user-defined parameter
[dR0dc] = solve_new_param_sensitivities_R0(x0,p,dR0dp,xdim,kdim);
if R0_only == 0
    fprintf('Sensitivity of R0 wrt new parameter at t=0 is %13.6e \n\n', dR0dc)
end

% Elasticity of R0 wrt user-defined parameter
[elR0c]=solve_new_param_elasticities_R0(r0,cparam,dR0dc,iplot);
if R0_only == 0
    fprintf('Elasticity of R0 wrt new parameter at t=0 is %13.6e \n\n', elR0c)
end

% Plot R0 information in a bar plot?
plot_R0(output_directory,r0,dR0dp,elR0p,dR0dc,elR0c,klist,iplot,R0_only)


end

