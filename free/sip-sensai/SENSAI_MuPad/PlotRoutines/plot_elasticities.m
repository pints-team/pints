function plot_elasticities(output_directory,imap,t,x,elxp,...
                           xdim,kdim,stype,...
                           iplot,ilist,klist)
%
%   ***   plot_elasticities(output_directory,imap,t,x,elxp,...
%                           xdim,kdim,stype,...
%                           iplot,ilist,klist)   ***
%

global NCOLUMNS

[nx,nt]=size(x);
if nt ~= length(t)
    fprintf('plot_elasticities:  Houston, we have a problem \n')
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
% Plot elasticities wrt parameters               % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(iplot.dxdp.var == 1 || iplot_default == 1)
    for i=1:nilist
        tmp(:,:)=elxp(ilist(i),:,:);
        
        ifig=5000+i;
        figure(ifig);
        
        for k=1:nklist
            if imap==1
                subplot(nrsubk,ncsubk,k), plot(t,tmp(klist(k),1:nt),'^m', 'LineWidth',2)
            else
                subplot(nrsubk,ncsubk,k), plot(t,tmp(klist(k),1:nt),'-m', 'LineWidth',2)
            end
            title(['Elasticity of x(',int2str(ilist(i)),') wrt p(',int2str(klist(k)),')'], 'FontSize', 12)
            xlabel('t', 'FontSize', 12)
            ylabel(['elast\_x(',int2str(ilist(i)),')\_p(',int2str(klist(k)),')'], 'FontSize', 12)
        end
        
        figname=['-f',num2str(ifig)];
        filename=[output_directory,'\elasticities_x',num2str(i),'_param.eps'];
        print(figname,'-depsc',filename)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot elasticities wrt initial conditions       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if stype == 3
    if(iplot.dxdz == 1)
        for i=1:nilist
            tmp(:,:)=elxp(ilist(i),:,:);
            
            ifig=6000+i;
            figure(ifig)
            for j=1:nilist
                if imap==1
                    subplot(nrsubx,ncsubx,j), plot(t,tmp(kdim+ilist(j),1:nt),'vk', 'LineWidth',2)
                else
                    subplot(nrsubx,ncsubx,j), plot(t,tmp(kdim+ilist(j),1:nt),'-k', 'LineWidth',2)
                end
                title(['Elasticity of x(',int2str(ilist(i)),') wrt z(',int2str(ilist(j)),')'], 'FontSize', 12)
                xlabel('t', 'FontSize', 12)
                ylabel(['elast\_ x(',int2str(ilist(i)),')\_z(',int2str(ilist(j)),')'], 'FontSize', 12)
            end
            
            figname=['-f',num2str(ifig)];
            filename=[output_directory,'\elasticities_x',num2str(i),'_ics.eps'];
            print(figname,'-depsc',filename)
        end
        
    end
end
 
clear tmp

% Plot elasticities of all variables for parameters in klist
if(iplot.dxdp.param == 1)
        
    for k=1:nklist
        tmp(1:xdim,1:nt)=elxp(1:xdim,klist(k),1:nt);    
        ifig=8000+k;
        figure(ifig)
        
        for i=1:nilist
             if imap==1
                 subplot(nrsubx,ncsubx,i), plot(t,tmp(ilist(i),1:nt),'dc','LineWidth',2)
             else
                  subplot(nrsubx,ncsubx,i), plot(t,tmp(ilist(i),1:nt),'-c','LineWidth',2)
             end
             title(['Elasticity of x(',int2str(ilist(i)),') wrt p(',int2str(klist(k)),')'], 'FontSize', 12)
             xlabel('t', 'FontSize', 12)
             ylabel(['elast\_x(',int2str(ilist(i)),')\_p(',int2str(klist(k)),')'], 'FontSize', 12)           
        end
        
       figname=['-f',num2str(ifig)];
       filename=[output_directory,'\elasticities_p',num2str(k),'.eps'];       
       print(figname,'-depsc',filename)     

    end
    
end
clear tmp

end
