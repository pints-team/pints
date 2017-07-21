function [input_directory,geom,Qvals,Ldim,nvol,ns,nq]=Input_Output_map(input_source,ExampleName,JobName)
%  ***   [input_directory,geom,Qvals,Ldim,nvol,ns,nq]=Input_Output_map(input_source,ExampleName,JobName)  ***
%
% Load input/output map from SENSAI_MuPad
%

input_directory=[input_source,'/',ExampleName,'/',JobName];
fprintf(['Input directory = ',input_directory, '\n']) 
load([input_directory,'/SIP_matfile.mat']);
%
fprintf('Number of input volumes       = %8i  \n', nvol)
%
nq=sum(ns);
fprintf('Number of forward simulations = %8i \n\n', nq)
%
fprintf('Qvals has %6i rows and %4i columns \n', size(Qvals,1),size(Qvals,2))
fprintf('Qdim = %4i \n\n', Qdim)

end