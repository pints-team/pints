function [Sdiag]=plot_active_subspace_singular_values(output_directory,nt,t,q,Mdim,Sdim,Fdim,S,sratio)
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
ifig=15200;
figure(ifig);

qmin=min(q);
qmax=max(q);
qrange=qmax-qmin;

for is=1:Sdim
    smax=max(Sdiag(is,2:nt));
    qscale=smax/qrange;
    subplot(nrsubk,ncsubk,is), plot(t,(q-qmin)*qscale,':k',t(2:nt),Sdiag(is,2:nt),'-.r','LineWidth',3)
    %title(['Active subspace: \lambda_',int2str(is)])
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', 16)
    if smax > 1E+03
        axis([0 t(nt) 0 1E+03])
    end
end
figname=['-f',num2str(ifig)];
filename=[output_directory,'/AS_singular_values_sensitivity_matrix_and_q.eps'];
print(figname,'-depsc',filename)


end

