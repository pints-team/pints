function gtype_SIP(output_directory,x0,p,tfinal,solntimes,ntsteps,xdim,kdim,qdim,...
                   qtype,stype,eFIM,qFIM,Fdim,Fp,iplot,ilist,klist)
%
%   ***  gtype_SIP(output_directory,x0,p,tfinal,ntsteps,xdim,kdim,qdim,...
%                  stype,eFIM,qFIM,Fdim,Fp,iplot,ilist,klist)   ***
%
%  Create binary input files for the stochastic inverse problem code SIP
%  by local sampling within the active subspace locally around each 
%  initial (random) point in the input (parameter) space.
%
%  Algorithm
%     Randomly sample the input space
%     Compute the leading (active) subspace of the sensitivity matrix at each point
%     Evaluate the map by locally sampling the leading subspace 
%  End
%
%  Variables
%    nvol    = number of volumes in parameter space
%    msample = number of samples of active subspace local to initial
%              (nsample) points
%    vmag  = radius of local active subspace sampled near each initial
%            (nvol) points 
%    pmin  = establishes lower bound of input (parameter) space
%    pmax  = establishes upper bound of input (parameter) space
%    qmin  = establishes lower bound of output space [SIP]
%    qmax  = establishes upper bound of output space [SIP]
%    cvar  = coefficient of variation of output distribution [SIP]
%    nedge = number of intervals in to which each dimension of the output
%            space is partitioned [SIP]
%    nvol  = number of locations at which the function is evaluated
%    nobs  = number of observations
%
%

iprint=2;
% Define (equally space) times at which the solution is to be computed                        
[yy,tt]=ode_setup(tfinal,ntsteps,solntimes,1,' ');

[ngrid,sgrid,msample,vmag,nobs,Qindex,nrsample,obserror,pmin,pmax,qmin,qmax,nsim,cvar,nedge]=user_SIP;
qdim=length(Qindex);
nvol=prod(ngrid);
msample=2*msample;   % To ensure msample is even

% if ntsteps == 0
%     nobs=1;
% else
%     nobs=ntsteps-1;
% end
psample=zeros(kdim,nvol*(msample+1));
qsample=zeros(qdim,nobs,nvol*(msample+1));

iactive=zeros(1,nvol);
Ustore=zeros(xdim,xdim,nvol);
Sstore=zeros(xdim,kdim,nvol);
Vstore=zeros(kdim,kdim,nvol);

% Compute sample solutions
% The primary samples are stored in the first nvol locations of the psample
% and qsample arrays
% We must test whether samples in the local subspaces lie outside the input space
% and reject them in these instances.
%

[pgrid,vgrid]=calculate_grid(ngrid,sgrid,kdim,pmin,pmax);
%plot(vgrid,'o:')

for ivol=1:nvol
    
    if mod(ivol,100)==0
        fprintf('\ngtype_SIP: Volume =  %5i \n', ivol)
    end
    
    % Generate a random sample in the input space
    % ptemp = pmin + (pmax-pmin).*rand(kdim,1); %uniformly distributed sample
    ptemp(1:kdim,1)=pgrid(1:kdim,ivol);
    
    psample(1:kdim,ivol) = ptemp;
    
    % Compute solution and sensitivities (depending upon value of stype)
    [nt,t,x,dxdp] = solve_ode(tt,x0,ptemp,xdim,kdim,stype);
    
    % Compute QoI and its sensitivities and elasticities
    % QoI is not a differentiable function of (x,p)
     if qtype == 0
         qsample(1:qdim,1:nrsample,ivol)=Qnondiff(t,x,ptemp,Qindex,nrsample,obserror);
         qdim=length(Qindex);
    end
    
    % QoI is a differentiable function of (x,p)
    if qtype == 1
        qsample(:,:,is)=Qsmartsample(output_directory,15,t,x,ptemp,x0,dxdp,...
                                     nt,xdim,kdim,qdim,stype,...
                                     ntsteps,msample,vmag,...
                                     iplot,ilist,klist)
    end
end

if iprint >= 4
    disp(psample)
    disp(qsample)
end

% Solve ode at center of parameter space to create upper and lower bounds
% on the output space
[nt,t,x,dxdp]=solve_ode(tt,x0,p,xdim,kdim,1);

pref=p;
if qtype == 0
    qref=Qnondiff(t,x,pref,Qindex,1,0);
end
if qtype == 1
    [q,dqdx,dqdparam]=solve_qoi(t,x,[p;x0],dxdp,xdim,kdim,qdim,1);
    if nobs==1
        qref=q(:,end);
    else
        qref=q;
    end
end

store_4_SIP=[output_directory,'/SENSAI_store.mat'];
fprintf('Saving SIP_store.mat \n')
save(store_4_SIP,...
    'Ustore','Sstore','Vstore','qdim','nobs',...
    'kdim','pmin','pmax',...
    'nvol','iactive','psample','qsample',...
    'nsim');

%Calculate volumes of Voronoi cells
if qtype==0
    psim=psample;
    qsim=qsample;
    vol=vgrid;
    nobs=1;
elseif qtype==1
    [psim,qsim,vol]=SIP_Voronoi(Ustore,Sstore,Vstore,qdim,nobs,...
                                kdim,pmin,pmax,...
                                nvol,iactive,psample,qsample,...
                                nsim);
end

% Write data for SIP
SIP_datafiles(output_directory,qdim,kdim, ...
              pmin,pmax,qmin,qmax, ...
              nobs,nvol,vol,nrsample,...
              psim,qsim, ...
              pref,qref,cvar,nedge);

% Write data for BET
% BET_datafiles(output_directory,qdim,kdim,pref,qref,...
%               nobs,nvol,psample,qsample)

% if Ldim==2
%     z=1;
% end
% save([output_directory,'input_output_basis.mat'],'x','y','z');


% Create reference solution to

end

