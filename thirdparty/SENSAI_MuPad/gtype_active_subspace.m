function [U,S,V]=gtype_active_subspace(output_directory,nt,t,x,p,dxdp,elxp, ...
                                       q,dqdparam,elqp, ...
                                       xdim,kdim,qdim, ...
                                       eFIM,qFIM,Fdim,Fp)
%
%  ***   [U,S,V]=gtype_active_subspace(output_directory,nt,t,x,p,dxdp,elxp,q,dqdparam,elqp,xdim,kdim,qdim, ...
%                              eFIM,qFIM,Fdim,Fp)   ***
%
%
%  Purpose
%  -------
%    Calculates the active subspace using sensitivities or elasticities of 
%    variables or QoIs
%
%  Variables
%  ---------
% Determine whether to construct the FIM from sensitivities or elasticities
% and whether to use primitive variables or quantities of interest
% eFIM = 0 -> sensitivities
%      = 1 -> elasticities
% qFIM = 0 -> primitive variables
%      = 1 -> quantities of interest
% Mdim = xdim  when using primitive variables
%      = qdim  when using quantities of interest
% Fdim = number of parameters to use to construct FIM
% Fp   = vector of length Fdim containing the parameters to use to
%        construct the FIM
%
%  Calls
%  -----
%  solve_active_subspace_angles
%  plot_active_subspace_singular_values
%  plot_active_subspace_singular_vectors
%  solve_active_subspace_approximation
%


% The sensitivity matrix M is of size Mdim x Fdim
% M = USV' where U is Mdim x Mdim
%                S is Mdim x Fdim
%                V is Fdim x Fdim
% Sdim is the minimum of Mdim and Fdim and is the number of singular values
%      Sdim = Mdim when Mdim < Fdim
%      Sdim = Fdim when Mdim > Fdim
%

iprint=0;

if eFIM==0
    if qFIM==0
        Mdim=xdim;
        dsdp=dxdp;
    elseif qFIM==1
        Mdim=qdim;
        dsdp=dqdparam;
    end
elseif eFIM==1
    if qFIM==0
        Mdim=xdim;
        dsdp=elxp;
    elseif qFIM==1
        Mdim=qdim;
        dsdp=elqp;
    end  
end 

M=zeros(Mdim,Fdim);
Sdim=min(Mdim,Fdim);
sdiag=zeros(Sdim);
sratio=zeros(Sdim);

fprintf('gtype_active_subspace, xdim = %5i, kdim = %5i, qdim = %5i \n', xdim, kdim, qdim)
fprintf('gtype_active_subspace, qFIM = %5i, eFIM = %5i \n', qFIM, eFIM)
fprintf('gtype_active_subspace, Mdim = %5i, Sdim = %5i, Fdim = %5i \n', Mdim, Sdim, Fdim)

Mstore=zeros(Mdim,Fdim,nt);
Ustore=zeros(Mdim,Mdim,nt);
Sstore=zeros(Mdim,Fdim,nt);
Vstore=zeros(Fdim,Fdim,nt);
sratio_store=zeros(Sdim,nt);
    
% Compute the sensitivity matrix and its SVD at each time
for it=2:nt   

    if iprint >= 2
        fprintf('Time, t(%4i) = %13.6e \n', it, t(it))
    end
    M(1:Mdim,1:Fdim)=dsdp(1:Mdim,Fp(1:Fdim),it);
    [rM,cM]=size(M);
    if rM ~= Mdim | cM ~= Fdim
        fprintf('gtype_active_subspace: Houston, we have a problem \n')
        return
    end
    if iprint >= 2
        fprintf('gtype_active_subspace: rank(M) = %5i, cond(M) = %13.6e \n', ...
                 rank(M,1.0E-08),cond(M))
    end

    % Compute the SVD of the sensitivity matrix
    [U,S,V]=svd(M);
    
    sdiag(1:Sdim)=diag(S(1:Sdim,1:Sdim));
    if iprint >= 2
        for is=1:Sdim       
           fprintf('gtype_active_subspace: sdiag(%5i) = %13.6e \n', is, sdiag(is))
        end
    end
    
    for is=1:Sdim-1
        if sdiag(is+1)>1e-16
            sratio(is)=sdiag(is)/sdiag(is+1);
            if iprint > 2
                fprintf('gtype_active_subspace:   sratio(%3i) = %13.6e \n', is, sratio(is))
            end
        else
            fprintf('gtype_active_subspace:   sratio(%3i) undefined \n', is)
        end
    end

    if iprint >= 4
        fprintf('Vectors spanning range of sensitivity matrix \n')
        disp(U)
        fprintf('Singular value matrix of sensitivity matrix \n')
        disp(S)
        fprintf('Vectors spanning domain of sensitivity matrix \n')
        disp(V)
    end
    
    Mstore(1:Mdim,1:Fdim,it)=M(1:Mdim,1:Fdim);
    Ustore(1:Mdim,1:Mdim,it)=U(1:Mdim,1:Mdim);
    Sstore(1:Mdim,1:Fdim,it)=S(1:Mdim,1:Fdim);
    Vstore(1:Fdim,1:Fdim,it)=V(1:Fdim,1:Fdim);
    sratio_store(1:Sdim-1,it)=sratio(1:Sdim-1);
         
end

% Calculate and plot angles between active subspaces
[nangle,ang1,ang2,ang3]=solve_active_subspace_angles(nt,t,Mdim,Sdim,Fdim,Vstore);
plot_active_subspace_singular_values(output_directory,nt,t,Mdim,Sdim,Fdim,Sstore,sratio_store);
plot_active_subspace_singular_vectors(output_directory,nt,t,Mdim,Sdim,Fdim,Vstore,nangle,ang1,ang2,ang3);

[Sdiag]=plot_active_subspace_singular_values_and_q(output_directory,nt,t,q,Mdim,Sdim,Fdim,Sstore,sratio_store);
plot_active_subspace_singular_vectors_and_q(output_directory,nt,t,q,Mdim,Sdim,Fdim,Sdiag,Vstore);

plot_M_and_q(output_directory,nt,t,q,Mstore,Sstore,Mdim,Sdim,Fdim)

% Calculate and plot active subspace approximation
% if xdim==1
%     solve_active_subspace_approximation(output_directory,nt,t,x,p,xdim,Mdim,Sdim,Fdim,Fp,Ustore,Sstore,Vstore)
% end

end



