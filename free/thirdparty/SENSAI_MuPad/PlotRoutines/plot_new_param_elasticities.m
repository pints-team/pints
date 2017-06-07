function plot_new_param_elasticities(output_directory,imap,t,elxc,elqc,xdim,kdim,ilist)
%
%   ***   plot_new_param_elasticities(output_directory,imap,t,elxc,elqc,xdim,kdim,ilist)   ***
%

global NCOLUMNS

[nx,nt]=size(elxc);
if nt ~= length(t)
    fprintf('plot_new_param_elasticities:  Houston, we have a problem \n')
end

nilist=length(ilist);
ncsubx=NCOLUMNS(nilist+1);
nrsubx=ceil((nilist+1)/ncsubx);    

    
ifig=21000;
figure(ifig)
for i=1:nilist    
    if imap==1
        subplot(nrsubx,ncsubx,i); plot(t, elxc(ilist(i),1:nt),'ob','LineWidth',2)
    else
        subplot(nrsubx,ncsubx,i); plot(t, elxc(ilist(i),1:nt),'-b','LineWidth',2)
    end
    title(['Elasticity of x(',int2str(ilist(i)),') wrt \zeta'], 'FontSize', 12)
    xlabel('t', 'FontSize', 12)
    ylabel(['elast\_x(',int2str(ilist(i)),')\_\zeta'], 'FontSize', 12)         
end

if imap==1
    subplot(nrsubx,ncsubx,nilist+1), plot(t, elqc,'dr','LineWidth',2)
else
    subplot(nrsubx,ncsubx,nilist+1), plot(t, elqc,'-r','LineWidth',2)
end
title(['Elasticity of QoI wrt \zeta'], 'FontSize', 12)
xlabel('t', 'FontSize', 12)
ylabel(['elast\_Q\_\zeta'], 'FontSize', 12)  

figname=['-f',num2str(ifig)];
filename=[output_directory,'\elasticities_cp.eps'];       
print(figname,'-depsc',filename)     


end
