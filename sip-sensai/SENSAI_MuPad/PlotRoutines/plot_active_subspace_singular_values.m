function plot_active_subspace_singular_values(output_directory,nt,t,Mdim,Sdim,Fdim,S,sratio)
%
%  ***   plot_active_subspace_singular_values(output_directory,nt,t,Mdim,Sdim,Fdim,S,sratio)   ***
%
%
%  Purpose
%  -------
%    Plot singular values of the sensitivity matrix
%
%  Variables
%  ---------
%


global NCOLUMNS

nklist=Sdim;
ncsubk=NCOLUMNS(nklist);
nrsubk=ceil(nklist/ncsubk);
Sdiag=zeros(Sdim,nt);
for is=1:Sdim
   Sdiag(is,1:nt)=S(is,is,1:nt);
end

% Plot eigenvalues of FIM
ifig=15000;
figure(ifig);
for is=1:Sdim
    subplot(nrsubk,ncsubk,is), plot(t(2:nt),Sdiag(is,2:nt),'-x'), title(['Active subspace: \lambda_',int2str(is)])
    smax=max(Sdiag(is,2:nt));
    if smax > 1E+03
        axis([0 t(nt) 0 1E+03])
    end
end
figname=['-f',num2str(ifig)];
filename=[output_directory,'\ActiveSubspace_singular_values_sensitivity_matrix.eps'];
print(figname,'-depsc',filename)

% Plot ratio of eigenvalues
if Sdim > 1
    ifig=ifig+1;
    figure(ifig);
    for is=1:Sdim-1
        subplot(nrsubk,ncsubk,is), plot(t(2:nt),sratio(is,(2:nt)),'-x'), title(['Active subspace: \lambda_',int2str(is),'/\lambda_',int2str(is+1)])
        maxratio=max(sratio(is,2:nt));
        if maxratio > 1E+04
            axis([0 t(nt) 0 1E+04])
        end
    end
    figname=['-f',num2str(ifig)];
    filename=[output_directory,'\ActiveSubspace_singular_value_ratios_sensitivity_matrix.eps'];
    print(figname,'-depsc',filename)
end



end

