function [q,dqdparam,elqp] = gtype(DIR,JOB,imap,x0,p,tfinal,solntimes,ntsteps,qdim, ...
                                   eFIM,qFIM,Fdim,Fp,iFtimes,pest,pert, ...
                                   iplot,ilist,klist, ...
                                   qtype,stype,NextGen,R0_only, ...
                                   ilambda,imu,nu,ds,nstep)
%
%  Purpose
%  -------
%    Treats the system as a general nonlinear iteration
%        x(t+1,theta) = g(t,x(t,theta),theta)
%
%
%  Inputs
%  ------
%    imap   = 0  -> sensitivity analysis for an ordinary differential equation
%           = 1  -> sensitivity analysis for an iterated map
%           = 2  -> arclength continuation for regular solutions
%           = 3  -> bifurcation from a trivial solution
%           = 4  -> limit point
%           = 5  -> Hopf bifurcation point
%           = 6  -> symmetry breaking bifurcation point  ???
%           = 13 -> step to upper branch from bifurcation from trivial solution  ???
%           = 15 -> sample map for BET computations
%    x0     = initial conditions  
%    p      = parameters
%    psi    = adjoint data
%    tfinal = final time
%    iplot  = plot level
%    ilist  = variables for which sensitivity information is required
%    klist  = parameters for which sensitivity information is required
%    iplot  = structure containing plotting information
%    qtype  = 0 -> non-differentiable function of (x,p)
%           = 1 -> differentiable function of (x,p)
%    stype  = Option for computing only the solutions without
%             sensitivities and elasticities, for faster computation
%           = 1 -> solution only
%           = 2 -> solution and sensitivities wrt parameters
%           = 3 -> solution and sensitivities wrt parameters and initial conditions
%
%  Variables
%  ---------
%    t runs from 1 to tfinal and is the time step
%    i runs from 1 to xdim and is the row index
%    j runs from 1 to xdim and is the column index
%    k runs from 1 to kdim and is the parameter index
%
%  Calls
%  -----
%    solve_ode 
%      or 
%    solve_map 
%      or 
%    solve_bifurcation
%
%    gtype_sensitivities - prints and plots sensitivities wrt parameters and initial conditions
%    gtype_elasticities  - calculates, prints and plots elasticities wrt parameters and initial conditions
%
%    gtype_qoi  - calculates, prints and plot the quantities of interest and their sensitivities 
%                 and elasticities wrt to parameters and initial conditions
%    gtype_cp   - calculates sensitivities and elasticities of variables and quantities of interest
%                 wrt a user defined parameter
%    gtype_R0   - calculates, prints and plots R0, its sensitivities and elasticities 
%                 wrt to parameters and initial conditions
%    gtype_FIM  - calculates the FIM using sensitivities or elasticities of variables or QoIs
%    gtype_GESS - performs generalized least squares parameter fitting
%

global NCOLUMNS
NCOLUMNS=[1  2*ones(1,3)  3*ones(1,2)  4*ones(1,6)  5*ones(1,8)  6*ones(1,10)  7*ones(1,12)  8*ones(1,14)  9*ones(1,16)];

xdim=length(x0);
kdim=length(p);

output_directory=[DIR,'/',JOB];
[s,mess,messid]=mkdir(output_directory);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%     Compute solutions (and sensitivities) according to imap             %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if imap == 0
    [yy,tt]=ode_setup(tfinal,ntsteps,solntimes,pert,' ');
    [nt,t,x,dxdp]=solve_ode(tt,x0,p,xdim,kdim,stype);
    plot(t,x(1,:))
elseif imap == 1
    [nt,t,x,dxdp]=solve_map(x0,p,tfinal,xdim,kdim,stype);
elseif imap >= 2 & imap <= 13
    solve_bifurcation(output_directory,x0,p,xdim,kdim,ilambda,imu,nu,ds,nstep,...
                      iplot,ilist,klist)
elseif imap == 15
    gtype_SIP(output_directory,x0,p,tfinal,solntimes,ntsteps,xdim,kdim,qdim,...
              qtype,stype,eFIM,qFIM,Fdim,Fp,iplot,ilist,klist)
    return
else
    fprintf('Error: imap value unknown \n')
    return
end

% Print and plot sensitivities of the variables wrt parameters and initial conditions
solution_sensitivity_output(output_directory,imap,t,x,dxdp,...
                            nt,xdim,kdim,iplot,ilist,klist,...
                            stype)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%     Sensitivity calculations                                            %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(stype >= 2)
    
    % Calculate, print and plot the elasticities of the variables wrt parameters and initial conditions
    [elxp]=gtype_elasticities(output_directory,imap,t,x,p,x0,dxdp,...
                              nt,xdim,kdim,stype,...
                              iplot,ilist,klist);
    
    
    % Calculate, print and plot the quantities of interest and their sensitivities and elasticities
    % wrt to parameters and initial conditions
    [q,dqdx,dqdparam,elqp] = gtype_qoi(output_directory,imap,t,x,p,x0,dxdp,...
                                       nt,xdim,kdim,qdim,stype,...
                                       iplot,ilist,klist);
    
    
    % Calculate sensitivities and elasticities of variables and quantities of interest
    % wrt a user defined parameter
    [cparam,dxdc,dqdc,elxc,elqc] = gtype_cp(output_directory,imap,t,x,p,x0,dxdp,q,dqdparam,...
                                            nt,xdim,kdim,qdim,iplot,ilist,klist);
    
    
    % Calculate, print and plot R0 and its sensitivities and elasticities
    % wrt parameters and initial conditions
    if NextGen(1) ~= 0
        [r0,dR0dp,R0warnings] = gtype_R0(output_directory,imap,t,x,p,dxdp,cparam,...
                                         nt,xdim,kdim,NextGen,R0_only,...
                                         iplot,ilist,klist)
    end
       
else
    
    %Display a warning if stype selected but NextGen indices are specified
    if stype == 1 && NextGen(1) ~= 0
        fprintf('Warning:  You have specified solution only but also to compute R0.  To save time, set NextGen = 0.\n');
        msgbox('Warning:  You have specified solution only but also to compute R0.  To save time, set NextGen = 0.');
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%     Parameter estimation calculations                                   %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if imap==0 & stype >= 2

    if Fdim>0
        % Calculate the active subspaces using sensitivities or elasticities of variables or QoIs
        fprintf('\n****************************** \n')
        fprintf('Computing the active subspaces \n')
        fprintf('****************************** \n\n')
        [U,S,V]=gtype_active_subspace(output_directory,nt,t,x,[p;x0],dxdp,elxp,q,dqdparam,elqp, ...
                                      xdim,kdim,qdim, ...
                                      eFIM,qFIM,Fdim,Fp)
    end
     
    Fisher=1; 
    if Fisher==1
        if Fdim>0 & ntsteps==0
            % Calculate the FIM using sensitivities or elasticities of variables or QoIs
            fprintf('\n************************************************ \n')
            fprintf('Computing the FIM and parameter selection scores \n')
            fprintf('************************************************ \n\n')
            [U,S,V]=gtype_FIM(output_directory,nt,t,x,[p;x0],dxdp,elxp,q,dqdparam,elqp, ...
                              xdim,kdim,qdim, ...
                              eFIM,qFIM,Fdim,Fp)
        end
        
        if Fdim>0 & ntsteps > 0
            % Calculate the FIM using sensitivities or elasticities of variables or QoIs
            fprintf('\n************************************************ \n')
            fprintf('Computing the global FIM  \n')
            fprintf('************************************************ \n\n')
            [U,S,V]=gtype_FIM_global(output_directory,nt,t,x,[p;x0],dxdp,elxp,q,dqdparam,elqp, ...
                                     xdim,kdim,qdim, ...
                                     eFIM,qFIM,Fdim,Fp,iFtimes)
        end
    end

    if pest==1
        % Perform generalized least squares parameter fitting
        fprintf('\n********************* \n')
        fprintf('Estimating parameters \n')
        fprintf('********************* \n\n')
        gtype_GESS(DIR,t,yy,x0,p,xdim,kdim,qdim,qFIM,pert)
        return
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%     Save input data and solutions                                       %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Save to directory %s \n', output_directory)
fprintf('Saving the input data \n')
filename=[output_directory,'/inputs.txt'];
fid = fopen(filename, 'wt');
fprintf(fid,'\nInitial solution \n');
for ii = 1:xdim
    fprintf(fid, 'x0(%3i,1) = %13.6e \n', ii, x0(ii,1));
end

fprintf(fid,'\nParameters\n');
for k = 1:kdim
    fprintf(fid, 'p(%3i,1) = %13.6e \n', k, p(k,1));
end

fprintf(fid,'\nFinal time \n');
fprintf(fid, 'tfinal = %13.6e \n', tfinal);
fclose(fid);

fprintf('Saving the Matlab files \n')
copyfile('gvec.m',output_directory);
if(stype >= 2)
    copyfile('dgvec_dxvec.m',output_directory);
    copyfile('dgvec_dparam.m',output_directory);
    copyfile('qoi.m',output_directory);
    copyfile('cp.m',output_directory);
end

fprintf('Saving the solution \n')
filename=[output_directory,'/output.mat'];
if(stype >= 2)
    if(NextGen(1) ~= 0)
        save(filename, 't', 'x', 'p', 'dxdp', 'q', 'dqdparam', 'elxp', 'elqp', 'xdim', 'kdim', 'tfinal','cparam','dxdc','dqdc','elxc','elqc','r0','dR0dp','dR0dc','elR0p','elR0c')
    else
        save(filename, 't', 'x', 'p', 'dxdp', 'q', 'dqdparam', 'elxp', 'elqp', 'xdim', 'kdim', 'tfinal','cparam','dxdc','dqdc','elxc','elqc')
    end
else
    save(filename, 't', 'x', 'p', 'xdim', 'kdim', 'tfinal')
end



end






