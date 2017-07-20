function plot_samples_histogram(output_directory,edges,n,bin,...
                                Qindex,qdim,idim,iprint,iplot)
%
%   ***  plot_samples_histogram(output_directory,edges,n,bin,...
%                               Qindex,qdim,idim,iprint,iplot)  *** 
%
%  Plot histogram of simulation study
%  Called by SIP_SENSAI
%

fprintf('plot_samples_histogram: idim = %4i, qdim = %4i \n', idim, qdim)
if iprint >= 4     
    fprintf('plot_samples_histogram: n array \n')
    disp(n')
    fprintf('plot_samples_histogram: bin array \n')
    disp(bin')
end

figure(1)
subplot(1,qdim,idim), bar(edges,n,'histc')
title(['Simulations of QoI #',num2str(idim)],'FontSize',14)
xlabel('Value','FontSize',12)
ylabel('Count','FontSize',12)

ifig=9+idim;
figure(ifig);
bar(edges,n,'histc')
title(['Simulations of QoI #',num2str(Qindex(idim))],'FontSize',14)
xlabel('Value','FontSize',12)
ylabel('Count','FontSize',12)
figname=['-f',num2str(ifig)];
filename=[output_directory,'/Qsim',num2str(Qindex(idim)),'.eps'];
print(figname,'-depsc',filename)

end

