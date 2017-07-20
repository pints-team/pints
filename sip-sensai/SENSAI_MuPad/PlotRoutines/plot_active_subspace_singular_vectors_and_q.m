function plot_active_subspace_singular_vectors_and_q(output_directory,nt,t,q,Mdim,Sdim,Fdim,S,V)
%
%  ***   plot_active_subspace_singular_vectors(output_directory,nt,t,q,Mdim,Sdim,Fdim,S,V)   ***
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
if Fdim >= 2
    Vplot2(1:Fdim,1:nt)=V(1:Fdim,2,1:nt);
end
if Fdim >= 3
    Vplot3(1:Fdim,1:nt)=V(1:Fdim,3,1:nt);
end

% Plot components of the eigenvector, V_1
ifig=15299;
qmin=min(q);
qmax=max(q);
qrange=qmax-qmin;
qscale=1/qrange;

smin=min(S);
smax=max(S);
srange=smax-smin;
sscale=1/srange;

for k=1:nklist
    ifig=ifig+1;
    figure(ifig);
    line(t(2:nt),Vplot1(k,2:nt),'Color','b','LineStyle','-','LineWidth',3)       
    %title(['Active subspace: V1',int2str(k)])
    xlabel('t','FontSize',16)
    axis([0,t(nt),-1,1])
    ax1 = gca; % current axes
    ax1.LineWidth=2;
    ax1.XColor = 'k';
    ax1.XAxis.FontSize=16;
    ax1.YColor = 'b';
    ax1.YAxis.FontSize=16;
    
    ax1_pos = ax1.Position; % position of first axes
    %line(t,2*((q-qmin)*qscale-0.5),'Color','k','LineWidth',3)
    %line(t(2:nt),2*((S(2:nt)-smin)*sscale-0.5),'Color','r','LineWidth',3)
%     ax2 = axes('Position',ax1_pos,...
%                'XAxisLocation','top',   'XColor','r',...
%                'YAxisLocation','right', 'YColor','r',...
%                'Color','none'); 
    ax2 = axes('Position',ax1_pos,...
               'YAxisLocation','right', 'YColor','r', ...
               'Color','none'); 
    line(t(2:nt),S(2:nt),'Parent',ax2,'Color','r','LineStyle','-.','LineWidth',2)
    axis([0,t(nt),0,35])
    ax2.LineWidth=2;
    ax2.XAxis.TickValues=[];
    ax2.XAxis.Color='none';
    ax2.YAxis.TickValues=[0 10 20 30];
    %ax2.YAxis.TickLength=[0.02 0.02];
    %ax2.YAxis.TickDir='out';
    ax2.YAxis.FontSize=16;
    ax2.YAxis.Color='r';

    ax3 = axes('Position',ax1_pos,...
               'YAxisLocation','right', 'YColor','k',...
               'Color','none'); 
    line(t,q,'Parent',ax3,'Color','k','LineStyle','--','LineWidth',2)
    % axis([0,t(nt),-90 40]) %% For Hodgkin-Huxley plots only
    axis([0,t(nt),0 18]) %% For Logistci plots only
    %ax3.LineWidth=2;
    ax3.XAxis.TickValues=[];
    ax3.XAxis.Color='none';
    ax3.YAxis.TickValues=[];
    %ax3.YAxis.FontSize=16;
    ax3.YAxis.Color='none';
        
    figname=['-f',num2str(ifig)];
    filename=[output_directory,'/AS_eigenvector_sensitivity_matrix_V1_',num2str(k),'_and_q.eps'];
    print(figname,'-depsc',filename)
end

% Plot components of the eigenvector, V_2
if Fdim >= 2
    for k=1:nklist
        ifig=ifig+1;
        figure(ifig);
        plot(t,2*((q-qmin)*qscale-0.5),':r',t(2:nt),2*((S(2:nt)-smin)*sscale-0.5),'-.c',t(2:nt),Vplot2(k,2:nt),'-b','LineWidth',2)
        title(['Active subspace: V2',int2str(k)])
        axis([0,t(nt),-1,1])
        figname=['-f',num2str(ifig)];
        filename=[output_directory,'/AS_eigenvector_sensitivity_matrix_V2_',num2str(k),'_and_q.eps'];
        print(figname,'-depsc',filename)
    end
end

% Plot components of the eigenvector, V_3
if Fdim >= 3
    for k=1:nklist
        ifig=ifig+1;
        figure(ifig);
        plot(t,2*((q-qmin)*qscale-0.5),':r',t(2:nt),2*((S(2:nt)-smin)*sscale-0.5),'-.c',t(2:nt),Vplot3(k,2:nt),'-','LineWidth',2)
        title(['Active subspace: V3',int2str(k)])
        axis([0,t(nt),-1,1])
        figname=['-f',num2str(ifig)];
        filename=[output_directory,'/AS_eigenvector_sensitivity_matrix_V3_',num2str(k),'_and_q.eps'];
        print(figname,'-depsc',filename)
    end
end




end

