function [U,S,V]=gtype_FIM(output_directory,nt,t,x,p,dxdp,elxp, ...
                           q,dqdparam,elqp, ...
                           xdim,kdim,qdim, ...
                           eFIM,qFIM,Fdim,Fp)
%
%  ***   [U,S,V]=gtype_FIM(output_directory,nt,t,x,p,dxdp,elxp, ...
%                          q,dqdparam,elqp, ...
%                          xdim,kdim,qdim, ...
%                          eFIM,qFIM,Fdim,Fp)   ***
%
%  Purpose
%  -------
%  Calculates the sensitivity matrix and its singular values and singular 
%  vectors, the FIM and its eigenvalues and eigenvectors and the parameter
%  selection score using sensitivities or elasticities of variables or QoIs 
%  as a function of time. 
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
iprint=2;
iplot=2;
iFIM=1;
ipss=0;

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

fprintf('gtype_FIM, xdim = %5i, kdim = %5i, qdim = %5i \n', xdim, kdim, qdim)
fprintf('gtype_FIM, qFIM = %5i, eFIM = %5i \n', qFIM, eFIM)
fprintf('gtype_FIM, Mdim = %5i, Sdim = %5i, Fdim = %5i \n', Mdim, Sdim, Fdim)

Ustore=zeros(Mdim,Mdim,nt);
Sstore=zeros(Mdim,Fdim,nt);
Vstore=zeros(Fdim,Fdim,nt);

% Compute sensitivity matrix and FIM at each time
for it=2:nt
    
    if iprint >= 2
        fprintf('Time, t(%4i) = %13.6e \n', it, t(it))
    end
    M(1:Mdim,1:Fdim)=dsdp(1:Mdim,Fp(1:Fdim),it);
    [rM,cM]=size(M);
    if rM ~= Mdim | cM ~= Fdim
        fprintf('gtype_FIM: Houston, we have a problem \n')
        return
    end
    if iprint >= 2
        fprintf('gtype_FIM: rank(M) = %5i, cond(M) = %13.6e \n', ...
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
                fprintf('gtype_FIM:   sratio(%3i) = %13.6e \n', is, sratio(is))
            end
        else
            fprintf('gtype_FIM:   sratio(%3i) undefined \n', is)
        end
    end
    
    if iprint >= 4
        fprintf('Vectors spanning domain of sensitivity matrix \n')
        disp(U)
        fprintf('Singular value matrix of sensitivity matrix \n')
        disp(S)
        fprintf('Vectors spanning range of sensitivity matrix \n')
        disp(V)
    end
    
    Ustore(1:Mdim,1:Mdim,it)=U;
    Sstore(1:Mdim,1:Fdim,it)=S;
    Vstore(1:Fdim,1:Fdim,it)=V;
    sratio_store(1:Sdim-1,it)=sratio(1:Sdim-1);
     
    if iFIM==1
        % Construct FIM = inv(M'*M)
        if Mdim >= Fdim            
            MM=M'*M;
            if iprint >= 2
                fprintf('gtype_FIM: rank(MM) = %5i, cond(MM) = %13.6e \n', ...
                    rank(MM,1.0E-08),cond(MM))
            end
            FIM=inv(MM);
            [Qeig,Deig]=eig(FIM);
            for iF=1:Fdim
                fprintf('gtype_FIM:        eig(%3i)  of FIM = %13.6e \n', iF, Deig(iF,iF))
                fprintf('gtype_FIM:   sqrt(eig(%3i)) of FIM = %13.6e \n\n', iF, sqrt(Deig(iF,iF)))
                % fprintf('gtype_FIM: 1/sqrt(eig(%3i)) of FIM = %13.6e \n\n', iF, 1/sqrt(Deig(iF,iF)))
            end
            if iprint >= 2
                fprintf('Eigenvectors of FIM \n')
                disp(Qeig)
            end
        end     
    end
    
    if ipss==1
        % Compute the parameter selection score defined by Cintron-Arias et al
        pss(1:Fdim)=abs( sdiag.^2 ./ p(Fp)' );
        for iF=1:Fdim
            fprintf('gtype_FIM: pss(%3i) = %13.6e \n', Fp(iF), pss(iF))
        end
        fprintf('gtype_FIM: norm(pss) = %13.6e \n', norm(pss))
        fprintf('\n')
    end
     
end

if iplot >= 2
    
    % Plot the singular values of the sensitivity matrix
    ifig=16000;
    figure(ifig)
    hold on
    for i=1:Sdim
        splot(1,2:nt)=Sstore(i,i,2:nt);
        subplot(1,Sdim,i), plot(t(2:nt),splot(1,2:nt))
        title('Singular values of sensitivity matrix')
        xlabel('t')
        ylabel(['S(',num2str(i),')'])
    end
    figname=['-f',num2str(ifig)];
    filename=[output_directory,'\FIM_singular_values_sensitivity_matrix.eps'];
    print(figname,'-depsc',filename)
    
    ifig=ifig+1;
    figure(ifig)
    hold on
    for i=1:Sdim
        splot(1,2:nt)=Sstore(i,i,2:nt);
        plot(t(2:nt),splot(1,2:nt))
        title('Singular values of sensitivity matrix')
        xlabel('t')
        ylabel(['S(',num2str(i),')'])
    end
    figname=['-f',num2str(ifig)];
    filename=[output_directory,'\FIM_singular_values_sensitivity_matrix_subplots.eps'];
    print(figname,'-depsc',filename)
    
    % Plot the singular vectors of the sensitivity matrix
    for j=1:Fdim
        ifig=ifig+1;
        figure(ifig)
        for i=1:Fdim
            vplot(1,2:nt)=Vstore(i,j,2:nt);
            subplot(1,Fdim,i), plot(t(2:nt),vplot(2:nt))
            title(['Singular vector V(',num2str(j),')'])
            xlabel('t')
            ylabel(['V(',num2str(i),',',num2str(j),')'])
        end
        figname=['-f',num2str(ifig)];
        filename=[output_directory,'\FIM_singular_vector_sensitivity_matrix_',num2str(j)','.eps'];
        print(figname,'-depsc',filename)
    end
    
        % Plot the singular vectors of the sensitivity matrix
    for j=1:Fdim
        ifig=ifig+1;
        figure(ifig)
        hold on
        for i=1:Fdim
            vplot(1,2:nt)=Vstore(i,j,2:nt);
            plot(t(2:nt),vplot(2:nt))
            title(['Singular vector V(',num2str(j),')'])
            xlabel('t')
            ylabel(['V(',num2str(i),',',num2str(j),')'])
        end
        figname=['-f',num2str(ifig)];
        filename=[output_directory,'\FIM_singular_vector_sensitivity_matrix_',num2str(j)','_subplots.eps'];
        print(figname,'-depsc',filename)
    end
    
end

end

