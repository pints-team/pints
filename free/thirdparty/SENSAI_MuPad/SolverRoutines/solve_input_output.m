function gtype_SIP(output_directory,x0,p,tfinal,ntsteps,xdim,kdim,qdim,...
                   eFIM,qFIM,Fdim,Fp,iplot,ilist,klist)


iprint=4;
                        
% Define times at which the solution is to be computed                        
tt=linspace(0,tfinal,ntsteps);

[nsample,plower,pupper,msample,vmag,qlower,qupper,sig,nedge]=user_input_output;
pmin=plower*p;
pmax=pupper*p;

psample=zeros(kdim,nsample*msample);
qsample=zeros(qdim,(ntsteps-1),nsample*msample);
vstep=vmag*rand(1,msample);

% Compute sample solutions
for i=1:nsample
    
    fprintf('Solve_input_output: Sample number %5i \n', i)
    psample(1:kdim,i) = pmin + (pmax-pmin).*rand(kdim,1);
    %psample(1:kdim,i) = p + pstd.*randn(kdim,1);
    
    % Compute solution and sensitivities
    [nt,t,x,dxdp] = solve_ode(tt,x0,psample(:,i),xdim,kdim,0);
    
    % Compute elasticities
    [elxp]=gtype_elasticities(output_directory,15,t,x,p,x0,dxdp,...
        nt,xdim,kdim,iplot,ilist,klist);
    
    % Compute sensitivities and elasticities of QoI
    [q,dqdx,dqdparam,elqp] = gtype_qoi(output_directory,15,t,x,p,x0,dxdp,...
        nt,xdim,kdim,qdim,iplot,ilist,klist);
    
    % Save quantity of interest
    qsample(1:qdim,1:ntsteps,i)=q(1:qdim,1:ntsteps);
    
    % Compute SVD of global sensitivity matrix
    [U,S,V]=gtype_FIM_global(output_directory,nt,t,x,p,dxdp,elxp,...
                             q,dqdparam,elqp,xdim,kdim,qdim,...
                             eFIM,qFIM,Fdim,Fp);
    
    if iprint >= 4
        fprintf('Perturb parameters in direction \n')
        disp(V(:,1))
    end
    
    % Sample parameter space in leading subspace and recompute QoI
    for j=1:msample
        is=nsample+(i-1)*msample+j;
        psample(1:kdim,is) = psample(1:kdim,i) + vstep(1,j)*V(:,1);
        [nt,t,x,dxdp] = solve_ode(tt,x0,psample(:,is),xdim,kdim,1);
        [q,dqdx,dqdparam]=solve_qoi(t,x,[p;x0],dxdp,xdim,kdim,qdim);
        qsample(1:qdim,1:ntsteps,is)=q(1:qdim,1:ntsteps);
    end
    %pause
    
end

if iprint >= 4
    disp(psample)
    disp(qsample)
end

% Interface with SIP
Ldim=kdim;
Lmin=pmin';
Lmax=pmax';
pvol=prod(Lmax-Lmin);
nvol=nsample*(msample+1);
vol=pvol/nvol;
ns=ones(nvol,1);
for is=1:nvol
    geom(is,1:Ldim)=psample(1:Ldim,is);
    geom(is,Ldim+1)=vol;
end

Qdim=qdim*(ntsteps-1);

for is=1:nvol
    for it=2:ntsteps
        k=qdim*(it-2)+1;
        Qvals(is,k:k+qdim)=qsample(1:qdim,it-1,is);
    end
end



[nt,t,x,dxdp] = solve_ode(tt,x0,psample(:,i),xdim,kdim,1);
[q,dqdx,dqdparam]=solve_qoi(t,x,[p;x0],dxdp,xdim,kdim,qdim);

pref=p';
qref=q;
fmin=qlower*qref(2:ntsteps);
fmax=qupper*qref(2:ntsteps);
mean=qref(2:ntsteps);
sigma=sig*mean;
nedge=nedge*ones(1,Qdim);

save([output_directory,'/input_output.mat'],'Ldim','Lmin','Lmax','nvol','geom',...
                        'Qdim','ns','Qvals','fmin','fmax','mean','sigma','nedge');


% if Ldim==2
%     z=1;
% end
% save([output_directory,'input_output_basis.mat'],'x','y','z');

    
% Create reference solution to 


% fprintf('Saving pref.txt \n')
% fidp=fopen([output_directory,'/pref.txt'],'wt');
% for k=1:kdim
%     fprintf(fidp,'  %13.6e',p(k));
% end
% fprintf(fidp,'\n');
% fclose(fidp);
%
% fprintf('Saving qref.txt \n')
% fidq=fopen([output_directory,'/qref.txt'],'wt');
% for j=1:qdim
%     fprintf(fidq,'  %13.6e',qref(j));
% end
% fprintf(fidq,'\n');
% fclose(fidq);
%
% fprintf('Storing the map: nsample = %5i \n', nsample)
% fprintf('Saving samples.mat \n')
% filename=[output_directory,'/samples.mat'];
% save(filename,'psample','qsample')
%
% fprintf('Saving psamples.txt \n')
% fidp=fopen([output_directory,'/psamples.txt'],'wt');
% for i=1:nsample
%     for k=1:kdim
%         fprintf(fidp,'  %13.6e',psample(k,i));
%     end
%     fprintf(fidp,'\n');
% end
% fclose(fidp);
%
% fprintf('Saving qsamples.txt \n')
% fidq=fopen([output_directory,'/qsamples.txt'],'wt');
% for i=1:nsample
%     for j=1:qdim
%         fprintf(fidq,'  %13.6e',qsample(j,i));
%     end
%     fprintf(fidq,'\n');
% end
% fclose(fidq);


end

