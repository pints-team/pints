function [yy,tt]=ode_setup(tfinal,ntsteps,solntimes,pert,DIR)
%
%  ***   [yy,tt]=ode_setup(tfinal,ntsteps,solntimes,pert,DIR)   ***
%
%
% tfinal     -> solve on [0,tfinal]
% nsteps = 0 -> step size determined by ODE45
%        > 0 -> solve at ntsteps equally spaced points (including ics)
% pert   = 0 -> read experimental data for parameter fitting
%        ~=0 -> perturb computed data for parameter fitting
% DIR       -> directory containing experimental data for parameter fitting
%

if ntsteps == 0
    tt=[0 tfinal];
    yy=zeros(1,2);
elseif ntsteps > 0
    tt=solntimes;
    yy=zeros(1,ntsteps);
end

if pert == 0
    copyfile([DIR,'/experimental_data.m'],'experimental_data.m')
    [tt,yy]=experimental_data;
end

end

