function qsample = Qsmartsample(output_directory,itype,t,x,ptemp,x0,dxdp,...
    nt,xdim,kdim,qdim,stype,...
    ntsteps,msample,vmag,...
    iplot,ilist,klist)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% Compute elasticities
if stype >= 2
    [elxp]=gtype_elasticities(output_directory,15,t,x,ptemp,x0,dxdp,...
        nt,xdim,kdim,stype,...
        iplot,ilist,klist);
end

[q,dqdx,dqdparam,elqp] = gtype_qoi(output_directory,15,t,x,ptemp,x0,dxdp,...
    nt,xdim,kdim,qdim,stype,...
    iplot,ilist,klist);

% Store Quantity of Interest
if ntsteps == 0
    qsample(1:qdim,1,is)=q(1:qdim,end);
else
    qsample(1:qdim,1:nobs,is)=q(1:qdim,2:ntsteps);
end

% Compute SVD of global sensitivity matrix
if ntsteps == 0
    %         [U,S,V]=gtype_active_subspace(output_directory,nt,t,x,[ptemp;x0],dxdp,elxp,...
    %                                       q,dqdparam,elqp, ...
    %                                       xdim,kdim,qdim,...
    %                                       eFIM,qFIM,Fdim,Fp);
    [U,S,V]=gtype_FIM(output_directory,nt,t,x,[ptemp;x0],dxdp,elxp,...
        q,dqdparam,elqp, ...
        xdim,kdim,qdim,...
        eFIM,qFIM,Fdim,Fp);
else
    [U,S,V]=gtype_FIM_global(output_directory,nt,t,x,[ptemp;x0],dxdp,elxp,...
        q,dqdparam,elqp, ...
        xdim,kdim,qdim,...
        eFIM,qFIM,Fdim,Fp);
end
% Print singular vectors spanning active subspace
if iprint >= 2
    fprintf('gtype_SIP: Singular values \n')
    disp(S(1:qdim,1:qdim))
    fprintf('gtype_SIP: Singular vectors spanning active subspace \n')
    disp(V(:,1))
    %pause
end

Ustore(:,:,is)=U;
Sstore(:,:,is)=S;
Vstore(:,:,is)=V;

% Create random samples within the active subspace
% Compute and store quantity of interest
% vstep=vmag*(-1+2*rand(1,msample));

vstep=vmag*linspace(-1,1,msample);

for js=1:msample
    
    % Create sample in active subspace
    ptemp=psample(1:kdim,is) + vstep(1,js)*V(:,1);
    
    if abs(ptemp) > abs(pmin) & abs(ptemp) < abs(pmax)
        nvol=nvol+1;
        fprintf('gtype_SIP: is = %5i, js = %5i, nvol = %5i \n', is,js,nvol)
        psample(1:kdim,nvol)=ptemp;
        
        % Solve ode
        [nt,t,x,dxdp]=solve_ode(tt,x0,ptemp,xdim,kdim,1);
        
        % Compute Quantity of Interest
        [q,dqdx,dqdparam]=solve_qoi(t,x,[ptemp;x0],dxdp,xdim,kdim,qdim,1);
        
        % Store Quantity of Interest
        if ntsteps == 0
            qsample(1:qdim,1,nvol)=q(1:qdim,end);
        else
            qsample(1:qdim,1:nobs,nvol)=q(1:qdim,2:ntsteps);
        end
    else
        fprintf('gtype_SIP: Sample rejected is = %5i, js = %5i \n', is,js)
        tlower = ptemp-pmin > 0;
        tupper = pmax-ptemp > 0;
        tlower'
        tupper'
    end
end

% Last of the nodes in the active subspace for the ivol(th) Voronoi cell
iactive(is)=nvol;
end


