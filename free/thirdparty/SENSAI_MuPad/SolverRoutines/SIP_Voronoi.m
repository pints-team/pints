function [psim,qsim,vol]=SIP_Voronoi(Ustore,Sstore,Vstore,qdim,nobs,...
                                     kdim,pmin,pmax,...
                                     nsample,iactive,psample,qsample,...
                                     nsim)
%
% Inputs
%   Ustore(:,:,:) = U in svd of sensitivity matrix 
%   Sstore(:,:,:) = S in svd of sensitivity matrix 
%   Vstore(:,:,:) = V U in svd of sensitivity matrix 
%   qdim = dimension of QoI
%   nobs = number of times qdim is observed
%   kdim = number of parameters (<= dimension of input space)
%   pmin = minimum parameter values
%   pmax = maximum parameter values
%   nsample = number of nodes
%   iactive = index set of function evaluations in active subspaces
%   psample = parameter values 
%   qsample = QoI at psample locations
%   nsim = number of simulated values to be computed
%
% Output
%   psim = parameters at which QoI is approximated
%   qsim = approximate QoI at psim locations
%   vol = volumes of Voronoi cells

iplot=0;
z=zeros(kdim,1);   
w=zeros(kdim,1);
d=zeros(nsim,nsample);          % FIM distance between sample and every node
count=zeros(nsample+1,1);       % The number of samples in each Voronoi cell
sim2vol=zeros(nsim,1);          % Mapping from sample to Voronoi cell
vol2sim=zeros(nsample+1,nsim);  % Mapping from Voronoi cell to sample

alpha=0.9;

psim=zeros(kdim,nsim);
qsim=zeros(qdim,nobs,nsim);

% Choose nsim points in the parameter space
for isim=1:nsim
    psim(1:kdim,isim)=pmin(1:kdim)+rand(kdim,1).*(pmax-pmin);
end

% Assign each sample to a Voronoi cell corresponding to one of the node
% using the FIM metric
for isim=1:nsim
    p=psim(:,isim);
    for ivol=1:nsample;
        pnode=psample(:,ivol);      
        U=Ustore(:,:,ivol);
        S=Sstore(:,:,ivol);
        V=Vstore(:,:,ivol);
        % Calculate vector between isim(th) new point and ivol(th) node
        z=p-pnode;
        % Calculate components in local coordinate system defined by the
        % singular vectors
        w=V'*z;
        % Calculate weighted distance defined by singular values
        % alpha*(w'*S'*S*w) + (1-alpha)*w'*w
        d(isim,ivol) = alpha*(w'*S'*S*w) + (1-alpha)*w'*w;
    end
    [dmin(isim),kmin(isim)]=min(d(isim,:));
end

% Create a map from isim to ivol and from ivol to isim
max_distance=1.0E16;
for isim=1:nsim
    if dmin(isim)>max_distance
        fprintf('ismin = %8i: Distance exceeds maximum \n', isim)
        ivol=nsample+1;
    else
        ivol=kmin(isim);
    end
    sim2vol(isim)=ivol;
    count(ivol)=count(ivol)+1;
    vol2sim(ivol,count(ivol))=isim;
end
fprintf('sum(count)-nsim = %5i \n', sum(count)-nsim)

% Plot the Voronoi cells associated with each node
if iplot >= 2
    plot_Voronoi(pmin,pmax,psim,psample,nsample,count,vol2sim)
end

% Plot the assignment between samples and nodes
% Nodes are Voronoi cell centers plus uniform points in active subspace
if iplot >= 2
    ifig=56;
    figure(ifig)
    hold off; clf
    if kdim==3
        figure(57); hold off; clf
        figure(58); hold off; clf
        figure(59); hold off; clf
    end      
end

% Project samples on to active subspace and find the closest "node"
for ivol=1:nsample
    
    fprintf('SIP_Voronoi: ivol = %5i \n', ivol)
    % Calculate indices of the "smart" nodes in the active subspace
    if ivol==1
        i1=nsample+1;
    else
        i1=iactive(ivol-1)+1;
    end
    i2=iactive(ivol);
    nactive=i2-i1+1;
    
    % Calculate projection matrix into the active subspace
    V=Vstore(:,:,ivol);
    vv=V(:,1);
    P=vv*vv';
    
    % Find the closest node to every sample in the ivol(th) Voronoi cell
    for j=1:count(ivol)
        isim=vol2sim(ivol,j);
        p=psim(1:kdim,isim);
        pbase=psample(1:kdim,ivol);
        pp=pbase + P*(p-pbase);
        dsub=zeros(1,nactive+1);
        for ii=1:nactive
            dsub(ii)=norm(pp-psample(1:kdim,i1+ii-1),2);
        end
        dsub(nactive+1)=norm(pp-psample(1:kdim,ivol),2);
        
        % Assign the QoI at the sample point according to the value at
        % the nearest node
        [d,k]=min(dsub);
        if k==nactive+1
            pem(1:kdim,isim)=psample(1:kdim,ivol);
            for jobs=1:nobs
                j1=(jobs-1)*qdim+1;
                j2=j1+qdim-1;
                qsim(j1:j2,isim)=qsample(1:qdim,jobs,ivol);
            end
            fprintf('i=%3i, j=%3i, Assign base value %13.4e \n', ivol,j, qsample(ivol))
        else
            pem(1:kdim,isim)=psample(1:kdim,i1+k-1);
            for jobs=1:nobs
                j1=(jobs-1)*qdim+1;
                j2=j1+qdim-1;
                qsim(j1:j2,isim)=qsample(1:qdim,jobs,i1+k-1);
            end
            fprintf('i=%3i, j=%3i, Assign active subspace value %13.4e \n', ivol,j, qsample(i1+k-1))
        end
        
        if iplot >= 2
            if kdim==2
                plot_projections2active_2D(pmin,pmax,psim,pp,pem,psample,...
                    isim,ivol,i1,i2,nactive,ifig)
            elseif kdim==3
                plot_projections2active_3D(pmin,pmax,psim,pp,pem,psample,...
                    isim,ivol,i1,i2,nactive,ifig)
            end  
        end
        
    end
end
    
    % Calculate volumes
    Omega_vol=prod(pmax-pmin);
    %vol=Omega_vol/nsim *count;
    vol=Omega_vol/nsim
    
    % vol_Voronoi=Omega_vol/nsim*count;
    % for ivol=1:nvol+1
    %     fprintf('Voronoi cell %3i has volume %8.3e  \n', ivol, vol(ivol))
    % end
    
end


