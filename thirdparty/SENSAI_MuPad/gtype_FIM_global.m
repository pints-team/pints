function [U,S,V]=gtype_FIM_global(output_directory,nt,t,x,p,dxdp,elxp,...
                                  q,dqdparam,elqp,...
                                  xdim,kdim,qdim, ...
                                  eFIM,qFIM,Fdim,Fp,iFtimes)
%
%  ***   [U,S,V]=gtype_FIM_global(output_directory,nt,t,x,p,dxdp,elxp,...
%                                 q,dqdparam,elqp,...
%                                 xdim,kdim,qdim, ...
%                                 eFIM,qFIM,Fdim,Fp,iFtimes)   ***
%
%
%  Purpose
%  -------
%  Calculates the global sensitivity matrix, FIM and parameter selection score
%  using sensitivities or elasticities of variables or QoIs at fixed time points. 
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
% pss  = parameter selection score (see Cintron-Arias et al 2009)
%
%  Calls
%  -----
%


global NCOLUMNS
iprint = 2;
iplot = 2;

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

%Mdim_global=Mdim*(nt-1);
Mdim_global=Mdim*length(iFtimes);

M=zeros(Mdim_global,Fdim);

if iprint >= 2
    fprintf('gtype_FIM_global: qFIM = %2i, eFIM = %2i, kdim = %3i, Mdim = %3i, Mdim_global = %3i, Fdim = %3i \n\n',...
             qFIM, eFIM, kdim, Mdim, Mdim_global, Fdim)
end 

% Compute sensitivity matrix and FIM at each time
% for it=2:nt
%     ir1=(it-2)*Mdim+1;
%     ir2=(it-1)*Mdim;
%     M(ir1:ir2,1:Fdim)=dsdp(1:Mdim,Fp(1:Fdim),it);   
% end

for is=1:length(iFtimes);
    it=iFtimes(is);
    ir1=(is-1)*Mdim+1;
    ir2=is*Mdim;
    M(ir1:ir2,1:Fdim)=dsdp(1:Mdim,Fp(1:Fdim),it);   
end    

M
[rM,cM]=size(M);
if rM ~= Mdim_global | cM ~= Fdim
    fprintf('gtype_FIM_global: Houston, we have a problem \n')
    return
end

if iprint >= 2
    fprintf('gtype_FIM_global: rank(M) = %3i \n', rank(M,1.0E-08))
    fprintf('gtype_FIM_global: cond(M) = %13.6e \n\n', cond(M))
end

% Compute the SVD of the sensitivity matrix
[U,S,V]=svd(M);

Sdim=min(Mdim_global,Fdim);
sd(1,1:Sdim)=diag(S(1:Sdim,1:Sdim));
for is=1:Sdim
    sdinv(is)=1/(sd(is));
    fprintf('gtype_FIM_global: sd(%3i) = %13.6e \n', is, sd(is))
end
fprintf('\n')
   
for is=1:Sdim-1
    if sd(is+1)>1e-16
        sratio(is)=sd(is)/sd(is+1);
        fprintf('gtype_FIM_global:   sratio(%3i) = %13.6e \n', is, sratio(is))
    else
        fprintf('gtype_FIM_global:   sratio(%3i) undefined \n', is)
    end
end
fprintf('\n')

% fprintf('Matrix of singular vectors spanning range of M \n')
% disp(U)
% fprintf('Matrix of singular values \n')
% disp(S)
fprintf('gtype_FIM_global: Matrix of singular vectors spanning domain of M \n')
disp(V)   

% Construct FIM = inv(M'*M)
% if Mdim_global >= Fdim
%     MM=M'*M;
%     FIM=inv(MM);
%     [Qeig,Deig]=eig(FIM);
%     for iF=1:Fdim
%         fprintf('gtype_FIM_global:        eig(%3i)  of FIM = %13.6e \n',   iF, Deig(iF,iF))
%         fprintf('gtype_FIM_global: 1/sqrt(eig(%3i)) of FIM = %13.6e \n\n', iF, 1/sqrt(Deig(iF,iF)))
%     end
%     fprintf('Eigenvectors of FIM \n')
%     disp(Qeig)
% 
% %   Compute the parameter selection score
%     pss(1:Fdim)=abs( sdinv ./ p(Fp)' );
%     for iF=1:Fdim
%         fprintf('gtype_FIM_global:  pss(%3i) = %13.6e \n', Fp(iF), pss(iF))
%     end
%     fprintf('gtype_FIM_global: norm(pss) = %13.6e \n', norm(pss))
%     fprintf('\n')
% 
% end

% Plot the singular values of the sensitivity matrix

if iplot >= 2
    ifig=17000;
    figure(ifig);
    plot(1:Sdim,sd(1:Sdim),'o')
    title('Singular values of global sensitivity matrix')
    xlabel(['1:',num2str(Sdim)])
    ylabel(['S(1:',num2str(Sdim),')'])
    
    figname=['-f',num2str(ifig)];
    filename=[output_directory,'\singular_values_global_sensitivity_matrix.eps'];
    print(figname,'-depsc',filename)
    
    % Plot the singular vectors of the sensitivity matrix
    for j=1:Fdim
        figure(ifig+j);
        plot(1:Fdim,V(1:Fdim,j),'o')
        title(['Global singular vector V(',num2str(j),')'])
        xlabel(['1:',num2str(Sdim)])
        ylabel(['V(1:',num2str(Sdim),')'])
        figname=['-f',num2str(ifig+j)];
        filename=[output_directory,'\singular_vector_global_sensitivity_matrix_',num2str(j)','.eps'];
        print(figname,'-depsc',filename)
    end
end

end

