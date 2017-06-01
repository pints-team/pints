function plot_marginal_input_probabilities(output_directory,qdim,idim,jdim,pmin,pmax,edges,pnedge,ppsum)
%
%  ***   plot_marginal_input_probabilities(output_directory,qdim,idim,jdim,pmin,pmax,edges,pnedge,ppsum)  ***
%
%  Plot the computed marginal input probabilities
%  Called by SIP_SENSAI
%

ifig=99+idim;
figure(ifig);
subplot(1,qdim+1,jdim), plot(edges(idim,1:pnedge(idim)),ppsum(jdim,1:pnedge(idim)),'o-','LineWidth',2)
xlabel(['p(',num2str(idim),')'])
ylabel('Probability')
title(['Marginal probability for p(',num2str(idim),') using jdim=',num2str(jdim)])
axis([0.99*pmin(idim),1.01*pmax(idim),0,0.35])

figname=['-f',num2str(ifig)];
filename=[output_directory,'/p',num2str(idim),'.eps'];
print(figname,'-depsc',filename)

end

