function plot_input_probabilities(output_directory,pinp,ndim,idim,p,nvol)
%
%   ***  plot_input_probabilities(output_directory,pinp,ndim,idim,p,nvol)  ***
%
%  Plot computed input probabilities
%  Called by SIP_SENSAI
%

ifig=39+idim;
figure(ifig)
pinp_min=min(pinp);
pinp_max=max(pinp);
fprintf('plot_pinp: idim=%4i, min(pinp)=%13.6e, max(pinp)=%13.6e \n',idim,pinp_min,pinp_max)
if ndim==3
    scatter3(p(:,1),p(:,2),p(:,3),25,pinp,'*')
    title(['Input joint density: idim = ', num2str(idim), ', nvol = ',num2str(nvol)])
    xlabel('p(1)')
    ylabel('p(2)')
    zlabel('p(3)')
elseif ndim==2
    scatter(p(:,1),p(:,2),25,pinp,'*')
    %scatter(p(:,1),p(:,2),25,ones(length(p(:,1)),1),'*')
%     title(['Input joint density: idim = ', num2str(idim), ', nvol = ',num2str(nvol)])
%     xlabel('p(1)')
%     ylabel('p(2)')
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', 16)

end

set(gca,'Clim',[pinp_min,pinp_max])
%set(gca,'Clim',[0.9,1.1])
colorbar('location','eastoutside')
p1min=0.99*min(p(:,1));
p1max=1.01*max(p(:,1));
p2min=0.99*min(p(:,2));
p2max=1.01*max(p(:,2));
if ndim==2
    axis([p1min,p1max,p2min,p2max])
elseif ndim == 3
    p3min=0.999*min(p(:,3));
    p3max=1.001*max(p(:,3));
    axis([p1min,p1max,p2min,p2max,p3min,p3max])
end

% Save figure to output directory
figname=['-f',num2str(ifig)];
filename=[output_directory,'/q',num2str(idim),'.eps'];
print(figname,'-depsc',filename)


end

