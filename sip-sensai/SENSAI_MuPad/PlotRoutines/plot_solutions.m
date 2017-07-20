function plot_solutions(output_directory,imap,t,x,ilist)
%
%   ***   plot_solutions(output_directory,imap,t,x,ilist)   ***
%
%

global NCOLUMNS

[nx,nt]=size(x);
if nt ~= length(t)
    fprintf('plot_solutions: Houston, we have a problem \n')
end

nilist=length(ilist);
ncsubx=NCOLUMNS(nilist);
nrsubx=ceil(nilist/ncsubx);

% Plots solutions by default

ifig=1;
figure(ifig)
for i=1:nilist
    if imap == 1
        subplot(nrsubx,ncsubx,i),plot(t,x(ilist(i),1:nt),'dg', 'LineWidth',2)
    else
        subplot(nrsubx,ncsubx,i),plot(t,x(ilist(i),1:nt),'-g', 'LineWidth',2)
    end
    title(['x(', num2str(ilist(i)), ')'], 'FontSize', 12)
    xlabel('t', 'FontSize', 12)
    ylabel(['x(', num2str(ilist(i)), ')'], 'FontSize', 12)
end

figname=['-f',num2str(ifig)];
filename=[output_directory,'/solution.eps'];
print(figname,'-depsc',filename)


end
