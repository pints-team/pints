function plot_qoi_sensitivities(output_directory,imap,iq,t,q,dqdparam,...
                                xdim,kdim,stype,...
                                iplot,ilist,klist)
%
%   ***   plot_qoi_sensitivities(output_directory,imap,iq,t,q,dqdparam,...
%                                xdim,kdim,stype,...
%                                iplot,ilist,klist)   ***
%

global NCOLUMNS

nt=length(t);
nq=length(q);
if nt ~= nq
    fprintf('plot_qoi_sensitivities:  Houston, we have a problem \n')
end

nklist=length(klist);
ncsubk=NCOLUMNS(nklist+1);
nrsubk = ceil((nklist+1)/ncsubk);

nilist=length(ilist);
ncsubx=NCOLUMNS(nilist+1);
nrsubx=ceil((nilist+1)/ncsubx);    

% Plot sensitivity wrt parameters
if iplot.dqdp == 1
    ifig=9000+(iq-1);
    figure(ifig)
    if imap==1
        subplot(nrsubk,ncsubk,1), plot(t,q, 'or', 'LineWidth',2)
    else
        subplot(nrsubk,ncsubk,1), plot(t,q, '-r', 'LineWidth',2)
    end
    title(['QoI(',num2str(iq),')'], 'FontSize', 12)
    xlabel('t', 'FontSize', 12)
    ylabel('Q', 'FontSize', 12)
    
    for k=1:nklist
        if imap==1
            subplot(nrsubk,ncsubk,k+1), plot(t,dqdparam(klist(k),1:nt),'dg', 'LineWidth',2)
        else
            subplot(nrsubk,ncsubk,k+1), plot(t,dqdparam(klist(k),1:nt),'-g', 'LineWidth',2)
        end
        title(['d(QoI(',num2str(iq),'))/dp(',int2str(klist(k)),')'], 'FontSize', 12)
        xlabel('t', 'FontSize', 12)
        ylabel(['dQ/dp(',int2str(klist(k)),')'], 'FontSize', 12)           
    end
    
     figname=['-f',num2str(ifig)];
     filename=[output_directory,'\sensitivities_qoi_param.eps'];       
     print(figname,'-depsc',filename)     
end

% Plot sensitivity wrt initial conditions
if stype == 3
    if iplot.dqdz == 1
        ifig=10000+(iq-1);
        figure(ifig)
        if imap==1
            subplot(nrsubx,ncsubx,1), plot(t,q, 'or', 'LineWidth',2)
        else
            subplot(nrsubx,ncsubx,1), plot(t,q, '-r', 'LineWidth',2)
        end
        title(['QoI(',num2str(iq),')'], 'FontSize', 12)
        xlabel('t', 'FontSize', 12)
        ylabel('Q', 'FontSize', 12)
        
        for i=1:nilist
            if imap==1
                subplot(nrsubx,ncsubx,i+1), plot(t,dqdparam(kdim+ilist(i),1:nt),'dg', 'LineWidth',2)
            else
                subplot(nrsubx,ncsubx,i+1), plot(t,dqdparam(kdim+ilist(i),1:nt),'-g', 'LineWidth',2)
            end
            title(['d(QoI(',num2str(iq),'))/dz(',int2str(ilist(i)),')'], 'FontSize', 12)
            xlabel('t', 'FontSize', 12)
            ylabel(['dQ/dz(',int2str(ilist(i)),')'], 'FontSize', 12)
        end
        
        figname=['-f',num2str(ifig)];
        filename=[output_directory,'\sensitivities_qoi_ics.eps'];
        print(figname,'-depsc',filename)
        
    end
end


end

