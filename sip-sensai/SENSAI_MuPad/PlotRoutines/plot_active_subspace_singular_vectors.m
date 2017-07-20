function plot_active_subspace_singular_vectors(output_directory,nt,t,Mdim,Sdim,Fdim,V,nangle,ang1,ang2,ang3)
%
%  ***   plot_active_subspace_singular_vectors(output_directory,nt,t,Mdim,Sdim,Fdim,V,nangle,ang1,ang2,ang3)   ***
%
%
%  Purpose
%  -------
%    Plot singular vectors of the sensitivity matrix
%
%  Variables
%  ---------
%

global NCOLUMNS

nklist=Fdim;
ncsubk=NCOLUMNS(nklist);
nrsubk=ceil(nklist/ncsubk);

% Extract three leading singular vectors
Vplot1(1:Fdim,1:nt)=V(1:Fdim,1,1:nt);
if nangle >= 2
    Vplot2(1:Fdim,1:nt)=V(1:Fdim,2,1:nt);
end
if nangle >= 3
    Vplot3(1:Fdim,1:nt)=V(1:Fdim,3,1:nt);
end

% Plot components of the eigenvector, V_1
ifig=15100;
figure(ifig);
for k=1:nklist
    subplot(nrsubk,ncsubk,k), plot(t(2:nt),Vplot1(k,2:nt),'-x'), title(['Active subspace: V1',int2str(k)])
end
figname=['-f',num2str(ifig)];
filename=[output_directory,'\ActiveSubspace_eigenvector_sensitivity_matrix_V1.eps'];
print(figname,'-depsc',filename)


% Plot components of the eigenvector, V_2
if nangle >= 2
    ifig=ifig+1;
    figure(ifig);
    for k=1:nklist
        subplot(nrsubk,ncsubk,k), plot(t(2:nt),Vplot2(k,2:nt),'-x'), title(['Active subspace: V2',int2str(k)])
    end
    figname=['-f',num2str(ifig)];
    filename=[output_directory,'\ActiveSubspace_eigenvector_sensitivity_matrix_V2.eps'];
    print(figname,'-depsc',filename)
end

% Plot components of the eigenvector, V_3
if nangle >= 3
    ifig=ifig+1;
    figure(ifig);
    for k=1:nklist
        subplot(nrsubk,ncsubk,k), plot(t(2:nt),Vplot3(k,2:nt),'-x'), title(['Active subspace: V3',int2str(k)])
    end
    figname=['-f',num2str(ifig)];
    filename=[output_directory,'\ActiveSubspace_eigenvector_sensitivity_matrix_V3.eps'];
    print(figname,'-depsc',filename)
end


% Plot angles between subspaces
ifig=ifig+1;
rd=180/pi;
figure(ifig)
plot(t(3:nt),rd*ang1(1,3:nt),'-r','LineWidth',2)
title('Active subspace: Angle between subspaces spanned by Q_1')
xlabel('Time','FontSize',12)
figname=['-f',num2str(ifig)];
filename=[output_directory,'\ActiveSubspace_angle_subspaces_Q1.eps'];
print(figname,'-depsc',filename)

if nangle >= 2
    ifig=ifig+1;
    figure(ifig)
    plot(t(3:nt),rd*ang2(2,3:nt),'-r',t(3:nt),rd*ang2(1,3:nt),'-g','LineWidth',2)
    title('Active subspace: Angle between subspaces spanned by Q_1 and Q_2')
    xlabel('Time','FontSize',12)
    figname=['-f',num2str(ifig)];
    filename=[output_directory,'\ActiveSubspace_angle_subspaces_Q1_Q2.eps'];
    print(figname,'-depsc',filename)
end

if nangle >= 3
    ifig=ifig+1;
    figure(ifig)
    plot(t(3:nt),rd*ang3(3,3:nt),'-r',t(3:nt),rd*ang3(2,3:nt),'-g',t(3:nt),rd*ang3(1,3:nt),'-b','LineWidth',2)
    title('Active subspace: Angle between subspaces spanned by Q_1, Q_2 and Q_3')
    xlabel('Time','FontSize',12)
    figname=['-f',num2str(ifig)];
    filename=[output_directory,'\ActiveSubspace_angle_subspaces_Q1_Q2_Q3.eps'];
    print(figname,'-depsc',filename)
end



end

