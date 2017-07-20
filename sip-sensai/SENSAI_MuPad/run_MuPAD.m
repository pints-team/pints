function run_MuPAD(userdirectory)
%
%  ***   run_MuPAD(userdirectory)   ***
%
% Creates Matlab files from user_equations.
% Calls mm_interface which calls MuPAD routines
%

fprintf('\n\n********* \n')
fprintf('RunMuPAD \n')
fprintf('\n\n********* \n')

rehash path
addpath('MuPadRoutines',userdirectory)

fprintf('RunMuPAD: implement user equations \n')
[fparam,fvec]=user_equations;

% Note: user_inputs is called only to get solution_only
[DIR,JOB,imap,x0,param,tfinal,solntimes,ntsteps,qtype,stype,NextGen,R0_only]=user_inputs;
xdim=fparam(1);
kdim=fparam(2);

fprintf('RunMuPAD: implement user QoI \n')
[qoi,qdim]=user_QoI;
fparam(3)=qdim;

fprintf('RunMuPAD: implement user parameters \n')
[cp]=user_parameters;

fprintf('RunMuPAD: xdim  = %3i \n', xdim)
fprintf('RunMuPAD: kdim  = %3i \n', kdim)
for i=1:xdim
    fprintf('RunMuPAD: f(%3i)  = %s \n', i,fvec(i,:))
end
fprintf('RunMuPAD: cp  = %s \n', cp)

mm_interface(fparam,fvec,qoi,cp,qtype,stype,NextGen,x0,param,imap,R0_only)
pause(2);

fprintf('\n\n*********************************\n')
fprintf('Matlab files successfully created \n')
fprintf('*********************************\n\n')
msgbox('MATLAB files successfully created');

rmpath('MuPadRoutines',userdirectory)


end

