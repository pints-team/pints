function run_brute_force(idim,nr,rtype,nedge);
% Estimate forward sensitivities by brute force.
% nr = number of realizations
% rtype = 0 -> uniform
%       = 1 -> nromal
% nedge = number of edges for histograms
%

rehash path
addpath('SolverRoutines', 'BifnRoutines', 'PlotRoutines')
addpath('Examples/ODE_examples/HodgkinHuxley')
%
xdim=4;
kdim=3;
%
x0(1,1) = -75;
x0(2,1) = 0.05;
x0(3,1) = 0.6;
x0(4,1) = 0.325;
%
p0(1,1) = 120;
p0(2,1) = 36;
p0(3,1) = 0.3;
%
tfinal = 12;
ntsteps = 0;
%ntsteps=0;
%
qtype = 1;
stype = 3;
%


nss=5;
nfedge=nedge*ones(1,nss);
fq=zeros(nss,nedge);
fmin=zeros(1,nss);
fmax=zeros(1,nss);
fedges=zeros(nss,nedge);
fpout=zeros(nss,nedge);

[fq,fmin,fmax,fedges,fpout]=brute_force(x0,p0,tfinal,ntsteps,xdim,kdim,stype,...
                                        rtype,nr,idim,nss,nfedge);
fprintf('fmin \n')
disp(fmin)
fprintf('fmax \n')
disp(fmax)
fprintf('fedges \n')
disp(fedges)
fprintf('fpout \n')
disp(fpout)

output_filename=['simulated_output_p',num2str(idim)]
save(output_filename,'fq') 
