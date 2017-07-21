function plot_sensitivities(output_directory,imap,t,x,dxdp,xdim,kdim,iplot,ilist,klist)
%
%   ***   plot_sensitivities(output_directory,imap,t,x,dxdp,xdim,kdim,iplot,ilist,klist)   ***
%

global NCOLUMNS

[nx,nt]=size(x);
if nt ~= length(t)
    fprintf('plot_sensitivities:  Houston, we have a problem \n')
end

nklist=length(klist);
ncsubk=NCOLUMNS(nklist);
nrsubk=ceil(nklist/ncsubk);

nilist=length(ilist);
ncsubx=NCOLUMNS(nilist);
nrsubx=ceil(nilist/ncsubx);    
    
iplot_default = 0;
if(iplot.dxdp.true == 1 && iplot.dxdp.var == 0 && iplot.dxdp.param == 0)
    iplot_default = 1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot sensitivities wrt parameters              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(iplot.dxdp.var == 1 || iplot_default == 1)
    for i=1:nilist
        tmp(:,:)=dxdp(ilist(i),:,:);
        
        ifig=1000+i;
        figure(ifig);
        
        for k=1:nklist
            if imap==1
                subplot(nrsubk,ncsubk,k), plot(t,tmp(klist(k),1:nt),'^m', 'LineWidth',2)
            else
                subplot(nrsubk,ncsubk,k), plot(t,tmp(klist(k),1:nt),'-m', 'LineWidth',2)
            end
            title(['dx(',int2str(ilist(i)),')/dp(',int2str(klist(k)),')'], 'FontSize', 12)
            xlabel('t', 'FontSize', 12)
            ylabel(['dx(',int2str(ilist(i)),')/dp(',int2str(klist(k)),')'], 'FontSize', 12)
        end
        
        figname=['-f',num2str(ifig)];
        filename=[output_directory,'\sensitivities_x',num2str(i),'_param.eps'];
        print(figname,'-depsc',filename)
        
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot sensitivities wrt initial conditions      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(iplot.dxdz == 1)
    for i=1:nilist
        tmp(:,:)=dxdp(ilist(i),:,:);
        
        ifig=2000+i;
        figure(ifig)
        for j=1:nilist
            if imap==1
                subplot(nrsubx,ncsubx,j), plot(t,tmp(kdim+ilist(j),1:nt),'vk', 'LineWidth',2)
            else
                subplot(nrsubx,ncsubx,j), plot(t,tmp(kdim+ilist(j),1:nt),'-k', 'LineWidth',2)
            end
            title(['dx(',int2str(ilist(i)),')/dz(',int2str(ilist(j)),')'], 'FontSize', 12)
            xlabel('t', 'FontSize', 12)
            ylabel(['dx(',int2str(ilist(i)),')/dz(',int2str(ilist(j)),')'], 'FontSize', 12)
        end
        
        figname=['-f',num2str(ifig)];
        filename=[output_directory,'\sensitivities_x',num2str(i),'_ics.eps'];
        print(figname,'-depsc',filename)
    end
    
end

clear tmp

% Plot sensitivities of all variables for parameters in klist
if (iplot.dxdp.param == 1)
        
    for k=1:nklist
        tmp(1:xdim,1:nt)=dxdp(1:xdim,klist(k),1:nt);    
        ifig=4000+k;
        figure(ifig)
        
        for i=1:nilist
             if imap==1
                 subplot(nrsubx,ncsubx,i), plot(t,tmp(ilist(i),1:nt),'dc','LineWidth',2)
             else
                  subplot(nrsubx,ncsubx,i), plot(t,tmp(ilist(i),1:nt),'-c','LineWidth',2)
             end
             title(['dx(',int2str(ilist(i)),')/dp(',int2str(klist(k)),')'], 'FontSize', 12)
             xlabel('t', 'FontSize', 12)
             ylabel(['dx(',int2str(ilist(i)),')/dp(',int2str(klist(k)),')'], 'FontSize', 12)           
        end
        
       figname=['-f',num2str(ifig)];
       filename=[output_directory,'\sensitivities_p',num2str(k),'.eps'];       
       print(figname,'-depsc',filename)     

    end
    
end
clear tmp

end
