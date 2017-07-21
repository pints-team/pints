function run_gtype(userdirectory,imap)

rehash path

addpath('SolverRoutines', 'BifnRoutines', 'PlotRoutines')
fprintf('\n\n********* \n')
fprintf('RunMatlab \n')
fprintf('********* \n')

fprintf('Model defined in directory %s \n', userdirectory)
addpath(userdirectory)

[DIR,JOB,imap0,x0,param,tfinal,solntimes,ntsteps,qtype,stype,NextGen,R0_only]=user_inputs;
[qoi,qdim]=user_QoI;
[ilambda,imu,nu,ds,nstep]=user_bifndata;
[eFIM,qFIM,Fdim,Fp,iFtimes,pest,pert]=user_FIMdata;
[iplot,ilist,klist]=user_plotdata;

% Compute map/ode solution and its stability wrt parameters and initial conditions
gtype(DIR,JOB,imap,x0,param,tfinal,solntimes,ntsteps,qdim,...
      eFIM,qFIM,Fdim,Fp,iFtimes,pest,pert,...
      iplot,ilist,klist,...
      qtype,stype,NextGen,R0_only,...
      ilambda,imu,nu,ds,nstep);

rmpath(userdirectory)
rmpath('SolverRoutines', 'BifnRoutines', 'PlotRoutines')
clear all;

end
