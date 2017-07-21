function plot_2D_projection(output_directory,nvol,geom,pinp,pmin,pmax,pedges,pnedge,qdim,idim,jdim)
%  ***   plot_2D_projection(output_directory,nvol,geom,pinp,pmin,pmax,pedges,pnedge,qdim,idim,jdim)    ***
%
%
%  Plot 2D projection through posteriori probability space
%

[n1,bin1]=histc(geom(:,idim),pedges(idim,1:pnedge(idim)));
[n2,bin2]=histc(geom(:,jdim),pedges(jdim,1:pnedge(jdim)));

ppsum(1:qdim+1,1:pnedge(idim),1:pnedge(jdim))=zeros(qdim+1,pnedge(idim),pnedge(jdim));
for kdim=1:qdim+1
    for ivol=1:nvol
        %fprintf('jdim=%3i, ivol=%5i i=%3i, j=%3i \n',jdim,ivol,bin1(ivol),bin2(ivol))
        ppsum(kdim,bin1(ivol),bin2(ivol))=ppsum(kdim,bin1(ivol),bin2(ivol))+pinp(ivol,kdim);      
    end
end

ifig=200+idim+jdim;
figure(ifig)
p_plot(1:pnedge(idim),1:pnedge(jdim))=ppsum(qdim+1,1:pnedge(idim),1:pnedge(jdim));
[nrp,ncp]=size(p_plot);
np=nrp*ncp;
p1=repmat(pedges(idim,1:pnedge(idim))',1,ncp);
p2=repmat(pedges(jdim,1:pnedge(jdim)), nrp,1);
pp1=reshape(p1,np,1);
pp2=reshape(p2,np,1);
pp_plot=reshape(p_plot,np,1);
scatter(pp1,pp2,40,pp_plot,'*') 
title(['p(',num2str(idim),') vs p(',num2str(jdim),')'],'Fontsize',14)
xlabel(['p(',num2str(idim),')'],'Fontsize',12)
ylabel(['p(',num2str(jdim),')'],'Fontsize',12)
figname=['-f',num2str(ifig)];
filename=[output_directory,'/p',num2str(idim),num2str(jdim),'.eps'];
print(figname,'-depsc',filename)




end

