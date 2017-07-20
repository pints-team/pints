function plot_new_param_sensitivities(output_directory,imap,t,dxdc,dqdc,xdim,kdim,ilist)
%
%   ***   plot_new_param_sensitivities(output_directory,imap,t,dxdc,dqdc,xdim,kdim,ilist)   ***
%

global NCOLUMNS

[nx,nt]=size(dxdc);
if nt ~= length(t)
    fprintf('plot_new_param_sensitivities:  Houston, we have a problem \n')
end

nilist=length(ilist);
ncsubx=NCOLUMNS(nilist+1);
nrsubx=ceil((nilist+1)/ncsubx);    

    
    ifig=20000;
    figure(ifig)
    for i=1:nilist    
        if imap==1
            subplot(nrsubx,ncsubx,i); plot(t, dxdc(ilist(i),1:nt),'ob','LineWidth',2)
        else
            subplot(nrsubx,ncsubx,i); plot(t, dxdc(ilist(i),1:nt),'-b','LineWidth',2)
        end
        title(['dx(',int2str(ilist(i)),')/d\zeta'], 'FontSize', 12)
        xlabel('t', 'FontSize', 12)
        ylabel(['dx(',int2str(ilist(i)),')/d\zeta'], 'FontSize', 12)         
    end
    
    if imap==1
        subplot(nrsubx,ncsubx,nilist+1), plot(t, dqdc,'dr','LineWidth',2)
    else
        subplot(nrsubx,ncsubx,nilist+1), plot(t, dqdc,'-r','LineWidth',2)
    end
    title(['d(QoI)/d\zeta'], 'FontSize', 12)
    xlabel('t', 'FontSize', 12)
    ylabel(['dQ/d\zeta'], 'FontSize', 12)  
    
    figname=['-f',num2str(ifig)];
    filename=[output_directory,'\sensitivities_cp.eps'];       
    print(figname,'-depsc',filename)     


end
